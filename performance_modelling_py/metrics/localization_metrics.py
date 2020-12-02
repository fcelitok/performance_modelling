#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import scipy
import time
import numpy as np
import pandas as pd
import os
from os import path
# from scipy.stats import t
import scipy.spatial
import subprocess

from performance_modelling_py.utils import print_info, print_error


def metric_evaluator(exec_path, poses_path, relations_path, weights, log_path, errors_path, unsorted_errors_path=None):
    with open(log_path, 'w') as stdout_log_file:
        if unsorted_errors_path is None:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path], stdout=stdout_log_file)
        else:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path, "-eu", unsorted_errors_path], stdout=stdout_log_file)
        p.wait()


def geometric_similarity(exec_path, scans_file_path, log_path, output_file_path, rate, range_limit):
    with open(log_path, 'w') as stdout_log_file:
        p = subprocess.Popen([exec_path, scans_file_path, output_file_path, str(rate), str(range_limit)], stdout=stdout_log_file)
        p.wait()


def compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df, average_rate=1.0, ground_truth_interpolation_tolerance=0.1):
    estimated_poses_df['t_datetime'] = pd.to_datetime(estimated_poses_df['t'], unit='s')
    ground_truth_poses_df['t_datetime'] = pd.to_datetime(ground_truth_poses_df['t'], unit='s')

    estimated_poses_df_len = len(estimated_poses_df['t'])
    estimated_poses_rate = estimated_poses_df_len/(estimated_poses_df['t'][estimated_poses_df_len-1] - estimated_poses_df['t'][0])
    estimated_poses_rate_limited_df = estimated_poses_df.iloc[::max(1, int(estimated_poses_rate/average_rate)), :]

    # interpolate ground_truth around estimate times
    tolerance = pd.Timedelta('{}s'.format(ground_truth_interpolation_tolerance))
    forward_matches = pd.merge_asof(
        left=estimated_poses_rate_limited_df[['t_datetime', 't', 'x', 'y', 'theta']],
        right=ground_truth_poses_df[['t_datetime', 't', 'x', 'y', 'theta']],
        on='t_datetime',
        direction='forward',
        tolerance=tolerance,
        suffixes=('_estimate', '_ground_truth'))
    backward_matches = pd.merge_asof(
        left=estimated_poses_rate_limited_df[['t_datetime', 't', 'x', 'y', 'theta']],
        right=ground_truth_poses_df[['t_datetime', 't', 'x', 'y', 'theta']],
        on='t_datetime',
        direction='backward',
        tolerance=tolerance,
        suffixes=('_estimate', '_ground_truth'))
    forward_backward_matches = pd.merge(
        left=backward_matches,
        right=forward_matches,
        on='t_estimate')

    interpolated_ground_truth_df = pd.DataFrame(columns=['t', 'x_est', 'y_est', 'theta_est', 'x_gt', 'y_gt', 'theta_gt'])
    for index, row in forward_backward_matches.iterrows():
        t_gt_1, t_gt_2 = row['t_ground_truth_x'], row['t_ground_truth_y']
        t_int = row['t_estimate']

        # if the estimate time is too far from a ground truth time (before or after), do not use this estimate data point
        if pd.isnull(t_gt_1) or pd.isnull(t_gt_2):
            continue

        x_est = row['x_estimate_x']
        x_gt_1, x_gt_2 = row['x_ground_truth_x'], row['x_ground_truth_y']
        x_int = np.interp(t_int, [t_gt_1, t_gt_2], [x_gt_1, x_gt_2])

        y_est = row['y_estimate_x']
        y_gt_1, y_gt_2 = row['y_ground_truth_x'], row['y_ground_truth_y']
        y_int = np.interp(t_int, [t_gt_1, t_gt_2], [y_gt_1, y_gt_2])

        theta_est = row['theta_estimate_x']
        theta_gt_1, theta_gt_2 = row['theta_ground_truth_x'], row['theta_ground_truth_y']
        theta_int = np.interp(t_int, [t_gt_1, t_gt_2], [theta_gt_1, theta_gt_2])

        interpolated_ground_truth_df = interpolated_ground_truth_df.append({
            't': t_int,
            'x_est': x_est,
            'y_est': y_est,
            'theta_est': theta_est,
            'x_gt': x_int,
            'y_gt': y_int,
            'theta_gt': theta_int,
        }, ignore_index=True)

    return interpolated_ground_truth_df


def lidar_visibility_environment_metric_for_each_waypoint(scans_file_path, run_events_file_path, range_limit, scans_rate=1.0):

    # check required files and directories exist
    if not path.isfile(scans_file_path):
        print_error("lidar_visibility_environment_metric_for_each_waypoint: scans file not found {}".format(scans_file_path))
        return

    if not path.isfile(run_events_file_path):
        print_error("lidar_visibility_environment_metric_for_each_waypoint: run_events file not found {}".format(run_events_file_path))
        return

    # get waypoints time intervals and compute the mean lidar visibility
    run_events_df = pd.read_csv(run_events_file_path, engine='python', sep=', ')
    target_pose_reached_df = run_events_df[run_events_df.event == 'target_pose_reached']
    target_pose_reached_timestamps_df = target_pose_reached_df.timestamp.reset_index(drop=True)
    waypoints_start_end_times_df = pd.DataFrame({'start_timestamp': list(target_pose_reached_timestamps_df[0:-1]), 'end_timestamp': list(target_pose_reached_timestamps_df[1:])})
    metric_results_per_waypoint = dict()
    metric_results_per_waypoint_list = list()

    # read scans file
    with open(scans_file_path) as scans_file:
        scan_lines = scans_file.read().split('\n')

    lidar_visibility_list = list()

    last_accepted_range_time = -np.inf
    for scan_line in scan_lines:
        scan_fields = scan_line.split(', ')
        if len(scan_fields) > 1:
            t, angle_min, angle_max, angle_increment, range_min, range_max = map(float, scan_fields[0:6])
            if t - last_accepted_range_time > 1.0/scans_rate:
                last_accepted_range_time = t
                ranges = list(map(float, scan_fields[6:]))
                visible_ranges = list(filter(lambda r: range_min < r < range_max and r < range_limit, ranges))
                num_valid_ranges = len(visible_ranges)
                record = {
                    't': t,
                    'ranges_min': float(min(ranges) if len(ranges) > 0 else np.nan),
                    'ranges_max': float(max(ranges) if len(ranges) > 0 else np.nan),
                    'ranges_median': float(np.median(ranges) if len(ranges) > 0 else np.nan),
                    'ranges_mean': float(np.mean(ranges) if len(ranges) > 0 else np.nan),
                    'visible_ranges_min': float(min(visible_ranges) if len(visible_ranges) > 0 else np.nan),
                    'visible_ranges_max': float(max(visible_ranges) if len(visible_ranges) > 0 else np.nan),
                    'visible_ranges_median': float(np.median(visible_ranges) if len(visible_ranges) > 0 else np.nan),
                    'visible_ranges_mean': float(np.mean(visible_ranges) if len(visible_ranges) > 0 else np.nan),
                    'visible_ranges_count': num_valid_ranges,
                    'visible_ranges_ratio': float(num_valid_ranges)/len(ranges),
                    'visible_fov_rad': float(num_valid_ranges * angle_increment),
                    'visible_fov_deg': float(num_valid_ranges * angle_increment * 180 / np.pi),
                }
                lidar_visibility_list.append(record)

    lidar_visibility_df = pd.DataFrame(lidar_visibility_list)

    # compute the lidar_visibility for each waypoint
    for index, start_timestamp, end_timestamp in waypoints_start_end_times_df[['start_timestamp', 'end_timestamp']].itertuples():
        assert(start_timestamp < end_timestamp)
        waypoint_scans_df = lidar_visibility_df.loc[(start_timestamp < lidar_visibility_df.t) & (lidar_visibility_df.t < end_timestamp)]
        waypoint_scans_dict = dict()
        waypoint_scans_dict["mean_ranges_min"] = float(waypoint_scans_df.ranges_min.mean())
        waypoint_scans_dict["mean_ranges_max"] = float(waypoint_scans_df.ranges_max.mean())
        waypoint_scans_dict["mean_ranges_median"] = float(waypoint_scans_df.ranges_median.mean())
        waypoint_scans_dict["mean_ranges_mean"] = float(waypoint_scans_df.ranges_mean.mean())
        waypoint_scans_dict["mean_visible_ranges_min"] = float(waypoint_scans_df.visible_ranges_min.mean())
        waypoint_scans_dict["mean_visible_ranges_max"] = float(waypoint_scans_df.visible_ranges_max.mean())
        waypoint_scans_dict["mean_visible_ranges_median"] = float(waypoint_scans_df.visible_ranges_median.mean())
        waypoint_scans_dict["mean_visible_ranges_mean"] = float(waypoint_scans_df.visible_ranges_mean.mean())
        waypoint_scans_dict["mean_visible_ranges_count"] = float(waypoint_scans_df.visible_ranges_count.mean())
        waypoint_scans_dict["mean_visible_ranges_ratio"] = float(waypoint_scans_df.visible_ranges_ratio.mean())
        waypoint_scans_dict["mean_visible_fov_deg"] = float(waypoint_scans_df.visible_fov_deg.mean())
        waypoint_scans_dict["start_time"] = start_timestamp
        waypoint_scans_dict["end_time"] = end_timestamp
        metric_results_per_waypoint_list.append(waypoint_scans_dict)

    metric_results_per_waypoint['version'] = "0.1"
    metric_results_per_waypoint['lidar_visibility_per_waypoint_list'] = metric_results_per_waypoint_list

    return metric_results_per_waypoint


def geometric_similarity_environment_metric_for_each_waypoint(log_output_folder, geometric_similarity_file_path, scans_file_path, run_events_file_path, scans_rate=1.0, range_limit=30.0, recompute=False):

    # check required files and directories exist
    if not path.exists(log_output_folder):
        os.makedirs(log_output_folder)

    if not path.isfile(scans_file_path):
        print_error("geometric_similarity_environment_metric_for_each_waypoint: scans file not found {}".format(scans_file_path))
        return

    if not path.isfile(run_events_file_path):
        print_error("geometric_similarity_environment_metric_for_each_waypoint: run_events file not found {}".format(run_events_file_path))
        return

    # compute the geometric_similarity for the whole run
    geometric_similarity_exec_path = path.join(path.dirname(path.abspath(__file__)), "cartographer_geometric_similarity")
    log_output_file_path = path.join(log_output_folder, "geometric_similarity_log.txt")
    if recompute or not path.exists(geometric_similarity_file_path):
        geometric_similarity(geometric_similarity_exec_path, scans_file_path, log_output_file_path, geometric_similarity_file_path, scans_rate, range_limit)

    # get waypoints time intervals and compute the mean geometric similarity
    run_events_df = pd.read_csv(run_events_file_path, engine='python', sep=', ')
    target_pose_reached_df = run_events_df[run_events_df.event == 'target_pose_reached']
    target_pose_reached_timestamps_df = target_pose_reached_df.timestamp.reset_index(drop=True)
    waypoints_start_end_times_df = pd.DataFrame({'start_timestamp': list(target_pose_reached_timestamps_df[0:-1]), 'end_timestamp': list(target_pose_reached_timestamps_df[1:])})
    metric_results_per_waypoint = dict()
    metric_results_per_waypoint_list = list()

    for index, start_timestamp, end_timestamp in waypoints_start_end_times_df[['start_timestamp', 'end_timestamp']].itertuples():
        assert(start_timestamp < end_timestamp)
        metric_results_per_waypoint_list.append(geometric_similarity_environment_metric(geometric_similarity_file_path, start_time=start_timestamp, end_time=end_timestamp))

    metric_results_per_waypoint['version'] = "0.4"
    metric_results_per_waypoint['geometric_similarity_per_waypoint_list'] = metric_results_per_waypoint_list

    return metric_results_per_waypoint


def geometric_similarity_environment_metric(geometric_similarity_file_path, start_time=None, end_time=None):

    # check required files exist
    if not path.isfile(geometric_similarity_file_path):
        print_error("compute_relative_localization_error: geometric_similarity file not found {}".format(geometric_similarity_file_path))
        return

    geometric_similarity_df = pd.read_csv(geometric_similarity_file_path)

    # if start and end times are specified drop the data outside the time range, otherwise get start and end times from the estimated poses
    if start_time is not None:
        geometric_similarity_df = geometric_similarity_df[geometric_similarity_df.t > start_time].reset_index(drop=True)
    else:
        start_time = geometric_similarity_df.iloc[0].t

    if end_time is not None:
        geometric_similarity_df = geometric_similarity_df[geometric_similarity_df.t < end_time].reset_index(drop=True)
    else:
        end_time = geometric_similarity_df.iloc[-1].t

    inf_covariance_count = len(geometric_similarity_df[geometric_similarity_df.x_x == np.inf])
    geometric_similarity_df = geometric_similarity_df[geometric_similarity_df.x_x < np.inf]  # ignore rows with infinite covariance (due to laser scan with no valid ranges)

    flat_covariance_mats = geometric_similarity_df[["x_x", "x_y", "x_theta", "y_x", "y_y", "y_theta", "theta_x", "theta_y", "theta_theta"]].values
    covariance_mats = flat_covariance_mats.reshape((len(flat_covariance_mats), 3, 3))

    metrics_result_dict = dict()

    if len(covariance_mats) > 0:
        metrics_result_dict['mean_of_covariance_x_x'] = float(np.mean(covariance_mats[:, 0, 0]))
        metrics_result_dict['mean_of_covariance_y_y'] = float(np.mean(covariance_mats[:, 1, 1]))
        metrics_result_dict['mean_of_covariance_theta_theta'] = float(np.mean(covariance_mats[:, 2, 2]))

        translation_eigenvalues, translation_eigenvectors = np.linalg.eig(covariance_mats[:, :2, :2])
        translation_eigenvalues_ratios = np.min(translation_eigenvalues, axis=1)/np.max(translation_eigenvalues, axis=1)
        metrics_result_dict['mean_of_translation_eigenvalues_ratio'] = float(1.0 - np.mean(translation_eigenvalues_ratios))
        metrics_result_dict['mean_of_translation_eigenvalues_ratio_all'] = float(1.0 - np.sum(translation_eigenvalues_ratios)/(len(translation_eigenvalues_ratios) + inf_covariance_count))
    else:
        metrics_result_dict['mean_of_translation_eigenvalues_ratio_all'] = 1.0

    metrics_result_dict['start_time'] = start_time
    metrics_result_dict['end_time'] = end_time
    metrics_result_dict['version'] = "0.3"
    return metrics_result_dict


def relative_localization_error_metrics_for_each_waypoint(log_output_folder, estimated_poses_file_path, ground_truth_poses_file_path, run_events_file_path, alpha=0.9, max_error=0.02, compute_sequential_relations=False):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("compute_relative_localization_error: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("compute_relative_localization_error: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    if not path.isfile(run_events_file_path):
        print_error("compute_relative_localization_error: run_events file not found {}".format(run_events_file_path))
        return

    run_events_df = pd.read_csv(run_events_file_path, engine='python', sep=', ')
    target_pose_reached_df = run_events_df[run_events_df.event == 'target_pose_reached']
    target_pose_reached_timestamps_df = target_pose_reached_df.timestamp.reset_index(drop=True)
    waypoints_start_end_times_df = pd.DataFrame({'start_timestamp': list(target_pose_reached_timestamps_df[0:-1]), 'end_timestamp': list(target_pose_reached_timestamps_df[1:])})
    metric_results_per_waypoint = dict()
    metric_results_per_waypoint_list = list()

    for index, start_timestamp, end_timestamp in waypoints_start_end_times_df[['start_timestamp', 'end_timestamp']].itertuples():
        assert(start_timestamp < end_timestamp)
        waypoint_log_output_folder_path = path.join(log_output_folder, "waypoint_{}".format(index))
        metric_results_per_waypoint_list.append(relative_localization_error_metrics(waypoint_log_output_folder_path, estimated_poses_file_path, ground_truth_poses_file_path, start_time=start_timestamp, end_time=end_timestamp))

    metric_results_per_waypoint['version'] = "0.1"
    metric_results_per_waypoint['relative_localization_error_per_waypoint_list'] = metric_results_per_waypoint_list
    metric_results_per_waypoint_translation_mean = 0
    metric_results_per_waypoint_rotation_mean = 0
    for relative_localization_error in metric_results_per_waypoint_list:
        try:
            metric_results_per_waypoint_translation_mean += relative_localization_error['random_relations']['translation']['mean']
            metric_results_per_waypoint_rotation_mean += relative_localization_error['random_relations']['rotation']['mean']
        except TypeError:  # in case translation or rotation was not computed
            print_error("relative_localization_error:", relative_localization_error)
            continue
    metric_results_per_waypoint['relative_localization_error_per_waypoint_mean'] = dict()
    metric_results_per_waypoint['relative_localization_error_per_waypoint_mean']['translation'] = metric_results_per_waypoint_translation_mean / len(metric_results_per_waypoint_list)
    metric_results_per_waypoint['relative_localization_error_per_waypoint_mean']['rotation'] = metric_results_per_waypoint_rotation_mean / len(metric_results_per_waypoint_list)
    return metric_results_per_waypoint


def relative_localization_error_metrics(log_output_folder, estimated_poses_file_path, ground_truth_poses_file_path, start_time=None, end_time=None, alpha=0.99, max_error=0.02, compute_sequential_relations=False):
    """
    Generates the ordered and the random relations files and computes the metrics
    """

    if not path.exists(log_output_folder):
        os.makedirs(log_output_folder)

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("compute_relative_localization_error: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("compute_relative_localization_error: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    relative_errors_dict = dict()

    # find the metricEvaluator executable
    metric_evaluator_exec_path = path.join(path.dirname(path.abspath(__file__)), "metricEvaluator", "metricEvaluator")

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)

    # if start and end times are specified drop the data outside the time range, otherwise get start and end times from the estimated poses
    if start_time is not None:
        estimated_poses_df = estimated_poses_df[estimated_poses_df.t > start_time].reset_index(drop=True)
        ground_truth_poses_df = ground_truth_poses_df[ground_truth_poses_df.t > start_time].reset_index(drop=True)
    else:
        start_time = estimated_poses_df.iloc[0].t

    if end_time is not None:
        estimated_poses_df = estimated_poses_df[estimated_poses_df.t < end_time].reset_index(drop=True)
        ground_truth_poses_df = ground_truth_poses_df[ground_truth_poses_df.t < end_time].reset_index(drop=True)
    else:
        end_time = estimated_poses_df.iloc[-1].t

    # compute the interpolated ground truth poses
    interpolated_ground_truth_df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)

    # if not enough matching ground truth data points are found, the metrics con not be computed
    if len(interpolated_ground_truth_df.index) < 2:
        print_error("no matching ground truth data points were found")
        return

    # convert estimated_poses_file to the CARMEN format
    estimated_poses_carmen_file_path = path.join(log_output_folder, "estimated_poses_carmen_format")
    with open(estimated_poses_carmen_file_path, "w") as estimated_poses_carmen_file:
        for index, row in estimated_poses_df.iterrows():
            estimated_poses_carmen_file.write("FLASER 0 0.0 0.0 0.0 {x} {y} {theta} {t}\n".format(x=row['x'], y=row['y'], theta=row['theta'], t=row['t']))

    # # random relations
    # relations_re_file_path = path.join(log_output_folder, "re_relations")
    # with open(relations_re_file_path, "w") as relations_file_re:
    #
    #     n_samples = 500
    #     for _ in range(n_samples):
    #         two_random_poses = interpolated_ground_truth_df[['t', 'x_gt', 'y_gt', 'theta_gt']].sample(n=2)
    #         t_1, x_1, y_1, theta_1 = two_random_poses.iloc[0]
    #         t_2, x_2, y_2, theta_2 = two_random_poses.iloc[1]
    #
    #         # reorder data so that t_1 is before t_2
    #         if t_1 > t_2:
    #             # swap 1 and 2
    #             t_1, x_1, y_1, theta_1, t_2, x_2, y_2, theta_2 = t_2, x_2, y_2, theta_2, t_1, x_1, y_1, theta_1
    #
    #         rel = get_matrix_diff((x_1, y_1, theta_1), (x_2, y_2, theta_2))
    #
    #         x_relative = rel[0, 3]
    #         y_relative = rel[1, 3]
    #         theta_relative = math.atan2(rel[1, 0], rel[0, 0])
    #
    #         relations_file_re.write("{t_1} {t_2} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(t_1=t_1, t_2=t_2, x=x_relative, y=y_relative, theta=theta_relative))
    #
    # print_info("\trelations", time.time() - relations_start, "s")
    #
    # # Run the metric evaluator on this relations file, read the sample standard deviation and exploit it to rebuild a better sample
    # metric_evaluator_start = time.time()
    #
    # # Compute translational sample size
    # summary_t_file_path = path.join(log_output_folder, "summary_t_errors")
    # metric_evaluator(exec_path=metric_evaluator_exec_path,
    #                  poses_path=estimated_poses_carmen_file_path,
    #                  relations_path=relations_re_file_path,
    #                  weights="{1, 1, 1, 0, 0, 0}",
    #                  log_path=path.join(log_output_folder, "summary_t.log"),
    #                  errors_path=summary_t_file_path)
    #
    # error_file = open(summary_t_file_path, "r")
    # content = error_file.readlines()
    # words = content[1].split(", ")
    # std = float(words[1])
    # print("\tstd", std, "is nan:", math.isnan(std))
    # z_a_2 = t.ppf(alpha, n_samples - 1)
    # print("\tz_a_2", z_a_2)
    # delta = max_error
    # n_samples_t = int(z_a_2**2 * std**2 / delta**2) if not math.isnan(std) else 0
    # print("\tn_samples_t", n_samples_t)
    #
    # # Compute rotational sample size
    # summary_r_file_path = path.join(log_output_folder, "summary_r_errors")
    # metric_evaluator(exec_path=metric_evaluator_exec_path,
    #                  poses_path=estimated_poses_carmen_file_path,
    #                  relations_path=relations_re_file_path,
    #                  weights="{0, 0, 0, 1, 1, 1}",
    #                  log_path=path.join(log_output_folder, "summary_r.log"),
    #                  errors_path=summary_r_file_path)
    # print_info("\tmetric_evaluator", time.time() - metric_evaluator_start, "s")
    #
    # error_file = open(summary_r_file_path, "r")
    # content = error_file.readlines()
    # words = content[1].split(", ")
    # std = float(words[1])
    # print("\tstd", std, "is nan:", math.isnan(std))
    # z_a_2 = t.ppf(alpha, n_samples - 1)
    # print("\tz_a_2", z_a_2)
    # delta = max_error
    # n_samples_r = int(z_a_2**2 * std**2 / delta**2) if not math.isnan(std) else 0
    # print("\tn_samples_r", n_samples_r)
    #
    # # Select the biggest of the two
    # n_samples = max(n_samples_t, n_samples_r)
    # if n_samples < 10:
    #     print_error("n_samples too low", n_samples, n_samples_t, n_samples_r)
    #     return

    relations_re_file_path = path.join(log_output_folder, "re_relations")
    n_samples = int(100*(end_time - start_time))

    interpolated_ground_truth_np = interpolated_ground_truth_df[['t', 'x_gt', 'y_gt', 'theta_gt']].values

    with open(relations_re_file_path, "w") as relations_file_re:
        for _ in range(n_samples):
            i_1, i_2 = np.random.choice(range(len(interpolated_ground_truth_df)), 2, replace=False)
            t_1, x_1, y_1, theta_1 = interpolated_ground_truth_np[i_1]
            t_2, x_2, y_2, theta_2 = interpolated_ground_truth_np[i_2]

            # reorder data so that t_1 is before t_2
            if t_1 > t_2:
                # swap 1 and 2
                t_1, x_1, y_1, theta_1, t_2, x_2, y_2, theta_2 = t_2, x_2, y_2, theta_2, t_1, x_1, y_1, theta_1

            rel = get_matrix_diff((x_1, y_1, theta_1), (x_2, y_2, theta_2))

            x_relative = rel[0, 3]
            y_relative = rel[1, 3]
            theta_relative = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_re.write("{t_1} {t_2} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(t_1=t_1, t_2=t_2, x=x_relative, y=y_relative, theta=theta_relative))

    relative_errors_dict['random_relations'] = dict()

    metric_evaluator_re_t_results_csv_path = path.join(log_output_folder, "re_t.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_re_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "re_t.log"),
                     errors_path=metric_evaluator_re_t_results_csv_path,
                     unsorted_errors_path=path.join(log_output_folder, "re_t_unsorted_errors"))

    metric_evaluator_re_t_results_df = pd.read_csv(metric_evaluator_re_t_results_csv_path, sep=', ', engine='python')
    relative_errors_dict['random_relations']['translation'] = dict()
    relative_errors_dict['random_relations']['translation']['mean'] = float(metric_evaluator_re_t_results_df['Mean'][0])
    relative_errors_dict['random_relations']['translation']['std'] = float(metric_evaluator_re_t_results_df['Std'][0])
    relative_errors_dict['random_relations']['translation']['min'] = float(metric_evaluator_re_t_results_df['Min'][0])
    relative_errors_dict['random_relations']['translation']['max'] = float(metric_evaluator_re_t_results_df['Max'][0])
    relative_errors_dict['random_relations']['translation']['n'] = float(metric_evaluator_re_t_results_df['NumMeasures'][0])

    metric_evaluator_re_r_results_csv_path = path.join(log_output_folder, "re_r.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_re_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "re_r.log"),
                     errors_path=metric_evaluator_re_r_results_csv_path,
                     unsorted_errors_path=path.join(log_output_folder, "re_r_unsorted_errors"))

    metric_evaluator_re_r_results_df = pd.read_csv(metric_evaluator_re_r_results_csv_path, sep=', ', engine='python')
    relative_errors_dict['random_relations']['rotation'] = dict()
    relative_errors_dict['random_relations']['rotation']['mean'] = float(metric_evaluator_re_r_results_df['Mean'][0])
    relative_errors_dict['random_relations']['rotation']['std'] = float(metric_evaluator_re_r_results_df['Std'][0])
    relative_errors_dict['random_relations']['rotation']['min'] = float(metric_evaluator_re_r_results_df['Min'][0])
    relative_errors_dict['random_relations']['rotation']['max'] = float(metric_evaluator_re_r_results_df['Max'][0])
    relative_errors_dict['random_relations']['rotation']['n'] = float(metric_evaluator_re_r_results_df['NumMeasures'][0])

    # sequential relations
    if compute_sequential_relations:
        ordered_relations_file_path = path.join(log_output_folder, "ordered_relations")
        with open(ordered_relations_file_path, "w") as relations_file_ordered:

            idx_delta = int(len(interpolated_ground_truth_df.index)/n_samples)
            if idx_delta == 0:
                idx_delta = 1

            for idx, first_row in interpolated_ground_truth_df.iloc[::idx_delta].iloc[0: -1].iterrows():
                second_row = interpolated_ground_truth_df.iloc[idx + idx_delta]

                rel = get_matrix_diff((first_row['x_gt'], first_row['y_gt'], first_row['theta_gt']), (second_row['x_gt'], second_row['y_gt'], second_row['theta_gt']))
                x_relative = rel[0, 3]
                y_relative = rel[1, 3]
                theta_relative = math.atan2(rel[1, 0], rel[0, 0])

                relations_file_ordered.write("{t_1} {t_2} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(t_1=first_row['t'], t_2=second_row['t'], x=x_relative, y=y_relative, theta=theta_relative))

        relative_errors_dict['sequential_relations'] = dict()

        metric_evaluator_ordered_t_results_csv_path = path.join(log_output_folder, "ordered_t.csv")
        metric_evaluator(exec_path=metric_evaluator_exec_path,
                         poses_path=estimated_poses_carmen_file_path,
                         relations_path=ordered_relations_file_path,
                         weights="{1, 1, 1, 0, 0, 0}",
                         log_path=path.join(log_output_folder, "ordered_t.log"),
                         errors_path=metric_evaluator_ordered_t_results_csv_path,
                         unsorted_errors_path=path.join(log_output_folder, "ordered_t_unsorted_errors"))

        metric_evaluator_ordered_t_results_df = pd.read_csv(metric_evaluator_ordered_t_results_csv_path, sep=', ', engine='python')
        relative_errors_dict['sequential_relations']['translation'] = dict()
        relative_errors_dict['sequential_relations']['translation']['mean'] = float(metric_evaluator_ordered_t_results_df['Mean'][0])
        relative_errors_dict['sequential_relations']['translation']['std'] = float(metric_evaluator_ordered_t_results_df['Std'][0])
        relative_errors_dict['sequential_relations']['translation']['min'] = float(metric_evaluator_ordered_t_results_df['Min'][0])
        relative_errors_dict['sequential_relations']['translation']['max'] = float(metric_evaluator_ordered_t_results_df['Max'][0])
        relative_errors_dict['sequential_relations']['translation']['n'] = float(metric_evaluator_ordered_t_results_df['NumMeasures'][0])

        metric_evaluator_ordered_r_results_csv_path = path.join(log_output_folder, "ordered_r.csv")
        metric_evaluator(exec_path=metric_evaluator_exec_path,
                         poses_path=estimated_poses_carmen_file_path,
                         relations_path=ordered_relations_file_path,
                         weights="{0, 0, 0, 1, 1, 1}",
                         log_path=path.join(log_output_folder, "ordered_r.log"),
                         errors_path=metric_evaluator_ordered_r_results_csv_path,
                         unsorted_errors_path=path.join(log_output_folder, "ordered_r_unsorted_errors"))

        metric_evaluator_ordered_r_results_df = pd.read_csv(metric_evaluator_ordered_r_results_csv_path, sep=', ', engine='python')
        relative_errors_dict['sequential_relations']['rotation'] = dict()
        relative_errors_dict['sequential_relations']['rotation']['mean'] = float(metric_evaluator_ordered_r_results_df['Mean'][0])
        relative_errors_dict['sequential_relations']['rotation']['std'] = float(metric_evaluator_ordered_r_results_df['Std'][0])
        relative_errors_dict['sequential_relations']['rotation']['min'] = float(metric_evaluator_ordered_r_results_df['Min'][0])
        relative_errors_dict['sequential_relations']['rotation']['max'] = float(metric_evaluator_ordered_r_results_df['Max'][0])
        relative_errors_dict['sequential_relations']['rotation']['n'] = float(metric_evaluator_ordered_r_results_df['NumMeasures'][0])

    relative_errors_dict['version'] = "0.1"
    relative_errors_dict['start_time'] = start_time
    relative_errors_dict['end_time'] = end_time

    return relative_errors_dict


def relative_localization_error_metrics_carmen_dataset(log_output_folder, estimated_poses_file_path, relations_file_path):
    """
    Generates the ordered and the random relations files and computes the metrics
    """

    if not path.exists(log_output_folder):
        os.makedirs(log_output_folder)

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("compute_relative_localization_error: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(relations_file_path):
        print_error("compute_relative_localization_error: relations file not found {}".format(relations_file_path))
        return

    # find the metricEvaluator executable
    metric_evaluator_exec_path = path.join(path.dirname(path.abspath(__file__)), "metricEvaluator", "metricEvaluator")

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return

    # convert estimated_poses_file to the CARMEN format
    estimated_poses_carmen_file_path = path.join(log_output_folder, "estimated_poses_carmen_format")
    with open(estimated_poses_carmen_file_path, "w") as estimated_poses_carmen_file:
        for index, row in estimated_poses_df.iterrows():
            estimated_poses_carmen_file.write("FLASER 0 0.0 0.0 0.0 {x} {y} {theta} {t}\n".format(x=row['x'], y=row['y'], theta=row['theta'], t=row['t']))

    # compute metrics
    relative_errors_dict = dict()
    relative_errors_dict['random_relations'] = dict()

    metric_evaluator_t_results_csv_path = path.join(log_output_folder, "t.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "t.log"),
                     errors_path=metric_evaluator_t_results_csv_path,
                     unsorted_errors_path=path.join(log_output_folder, "t_unsorted_errors"))

    metric_evaluator_t_results_df = pd.read_csv(metric_evaluator_t_results_csv_path, sep=', ', engine='python')
    relative_errors_dict['random_relations']['translation'] = dict()
    relative_errors_dict['random_relations']['translation']['mean'] = float(metric_evaluator_t_results_df['Mean'][0])
    relative_errors_dict['random_relations']['translation']['std'] = float(metric_evaluator_t_results_df['Std'][0])
    relative_errors_dict['random_relations']['translation']['min'] = float(metric_evaluator_t_results_df['Min'][0])
    relative_errors_dict['random_relations']['translation']['max'] = float(metric_evaluator_t_results_df['Max'][0])
    relative_errors_dict['random_relations']['translation']['n'] = float(metric_evaluator_t_results_df['NumMeasures'][0])

    metric_evaluator_r_results_csv_path = path.join(log_output_folder, "r.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "r.log"),
                     errors_path=metric_evaluator_r_results_csv_path,
                     unsorted_errors_path=path.join(log_output_folder, "r_unsorted_errors"))

    metric_evaluator_r_results_df = pd.read_csv(metric_evaluator_r_results_csv_path, sep=', ', engine='python')
    relative_errors_dict['random_relations']['rotation'] = dict()
    relative_errors_dict['random_relations']['rotation']['mean'] = float(metric_evaluator_r_results_df['Mean'][0])
    relative_errors_dict['random_relations']['rotation']['std'] = float(metric_evaluator_r_results_df['Std'][0])
    relative_errors_dict['random_relations']['rotation']['min'] = float(metric_evaluator_r_results_df['Min'][0])
    relative_errors_dict['random_relations']['rotation']['max'] = float(metric_evaluator_r_results_df['Max'][0])
    relative_errors_dict['random_relations']['rotation']['n'] = float(metric_evaluator_r_results_df['NumMeasures'][0])

    return relative_errors_dict


def get_matrix_diff(p1, p2):
    """
    Computes the rototranslation difference of two points
    """

    x1, y1, theta1 = p1
    x2, y2, theta2 = p2

    m_translation1 = np.matrix(((1, 0, 0, x1),
                                (0, 1, 0, y1),
                                (0, 0, 1, 0),
                                (0, 0, 0, 1)))

    m_translation2 = np.matrix(((1, 0, 0, x2),
                                (0, 1, 0, y2),
                                (0, 0, 1, 0),
                                (0, 0, 0, 1)))

    m_rotation1 = np.matrix(((math.cos(theta1), -math.sin(theta1), 0, 0),
                             (math.sin(theta1), math.cos(theta1), 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1)))

    m_rotation2 = np.matrix(((math.cos(theta2), -math.sin(theta2), 0, 0),
                             (math.sin(theta2), math.cos(theta2), 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1)))

    m1 = m_translation1 * m_rotation1
    m2 = m_translation2 * m_rotation2
    return m1.I * m2


def absolute_localization_error_metrics(estimated_poses_file_path, ground_truth_poses_file_path):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("absolute_localization_error_metrics: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("absolute_localization_error_metrics: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return

    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)
    df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)

    # if no matching ground truth data points are found, the metrics con not be computed
    if len(df.index) < 2:
        print_error("no matching ground truth data points were found")
        return

    absolute_error_of_each_segment = np.sqrt((df['x_est'] - df['x_gt'])**2 + (df['y_est'] - df['y_gt'])**2)

    absolute_error_dict = dict()
    absolute_error_dict['sum'] = float(absolute_error_of_each_segment.sum())
    absolute_error_dict['mean'] = float(absolute_error_of_each_segment.mean())
    absolute_error_dict['version'] = "0.1"
    return absolute_error_dict


def absolute_error_vs_voronoi_radius(estimated_poses_file_path, ground_truth_poses_file_path, ground_truth_map, samples_per_second=1.0):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("absolute_localization_error_metrics: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("absolute_localization_error_metrics: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return

    start_time = time.time()
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)

    ground_truth_interpolated_poses_file_path = path.splitext(estimated_poses_file_path)[0] + "ground_truth_interpolated.csv"
    if path.exists(ground_truth_interpolated_poses_file_path):
        complete_trajectory_df = pd.read_csv(ground_truth_interpolated_poses_file_path)
    else:
        complete_trajectory_df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)
        complete_trajectory_df.to_csv(ground_truth_interpolated_poses_file_path)

    run_duration = ground_truth_poses_df.iloc[-1]['t'] - ground_truth_poses_df.iloc[0]['t']
    resample_n = int(run_duration*samples_per_second)
    resample_ratio = max(1, len(complete_trajectory_df) // resample_n)
    trajectory_df = complete_trajectory_df.iloc[::resample_ratio].copy(deep=False)
    print_info("compute_ground_truth_interpolated_poses", time.time() - start_time)

    # if no matching ground truth data points are found, the metrics can not be computed
    if len(trajectory_df.index) < 2:
        print_error("no matching ground truth data points were found")
        return

    trajectory_points_list = list()
    trajectory_df['abs_err'] = np.sqrt((trajectory_df['x_est'] - trajectory_df['x_gt'])**2 + (trajectory_df['y_est'] - trajectory_df['y_gt'])**2)
    for _, row in trajectory_df.iterrows():
        x_gt, y_gt, abs_err = row['x_gt'], row['y_gt'], row['abs_err']
        trajectory_points_list.append(np.array([x_gt, y_gt]))

    start_time = time.time()
    min_radius = 4 * ground_truth_map.resolution
    voronoi_graph = ground_truth_map.voronoi_graph.subgraph(filter(
        lambda n: ground_truth_map.voronoi_graph.nodes[n]['radius'] >= min_radius,
        ground_truth_map.voronoi_graph.nodes
    ))
    print_info("ground_truth_map.voronoi_graph", time.time() - start_time)

    voronoi_vertices_list = list()
    voronoi_radii_list = list()
    for node_index, node_data in voronoi_graph.nodes.data():
        voronoi_vertices_list.append(node_data['vertex'])
        voronoi_radii_list.append(node_data['radius'])
    voronoi_vertices_array = np.array(voronoi_vertices_list)

    start_time = time.time()
    kdtree = scipy.spatial.cKDTree(voronoi_vertices_array)
    dist, indexes = kdtree.query(trajectory_points_list)
    print_info("kdtree", time.time() - start_time)

    trajectory_radii_list = list()
    for voronoi_vertex_index in indexes:
        trajectory_radii_list.append(voronoi_radii_list[voronoi_vertex_index])

    trajectory_df['trajectory_radius'] = trajectory_radii_list

    # # plotting
    # import matplotlib.pyplot as plt
    #
    # # fig, ax = plt.subplots()
    # # ax.scatter(trajectory_df['trajectory_radius'], trajectory_df['abs_err'])  # , color='red', s=4.0, marker='o')
    # # plt.xlabel("trajectory_radius")
    # # plt.ylabel("abs_err")
    # # plt.show()
    # # plt.cla()
    # #
    # # plt.plot(trajectory_df['t'], trajectory_df['abs_err'])
    # # plt.plot(trajectory_df['t'], trajectory_df['trajectory_radius'])
    # # plt.legend()
    # # plt.show()
    # # plt.cla()
    #
    # bins = np.linspace(0, 3, 4*5)
    # plt.hist(trajectory_df['abs_err'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['trajectory_radius'] > 0.0) & (trajectory_df['trajectory_radius'] < 3.5)]['abs_err'], bins=bins)
    # plt.title("0.0 ~ 3.5")
    # plt.show()
    # plt.cla()
    #
    # plt.hist(trajectory_df['abs_err'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['trajectory_radius'] > 3.5) & (trajectory_df['trajectory_radius'] < 5.0)]['abs_err'], bins=bins)
    # plt.title("3.5 ~ 5.0")
    # plt.show()
    # plt.cla()
    #
    # plt.hist(trajectory_df['abs_err'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['trajectory_radius'] > 0.0) & (trajectory_df['trajectory_radius'] < 1.0)]['abs_err'], bins=bins)
    # plt.title("0.0 ~ 1.0")
    # plt.show()
    # plt.cla()
    #
    # bins = np.linspace(0, 9, 8*4)
    # plt.hist(trajectory_df['trajectory_radius'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['abs_err'] > .3) & (trajectory_df['abs_err'] < .5)]['trajectory_radius'], bins=bins)
    # plt.show()
    #
    # # plot trajectory and distances from voronoi vertices
    # distance_segments = list()
    # for trajectory_point_index, voronoi_vertex_index in enumerate(indexes):
    #     voronoi_vertex = voronoi_vertices_array[voronoi_vertex_index]
    #     trajectory_point = trajectory_points_list[trajectory_point_index]
    #     distance_segments.append((voronoi_vertex, trajectory_point))
    #
    # trajectory_segments = zip(trajectory_points_list[0: -1], trajectory_points_list[1:])
    # ground_truth_map.save_voronoi_plot_and_trajectory("~/tmp/vg_and_traj.svg", trajectory_segments + distance_segments)

    return trajectory_df


def trajectory_length_metric(ground_truth_poses_file_path):

    # check required files exist
    if not path.isfile(ground_truth_poses_file_path):
        print_error("compute_trajectory_length: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return None

    df = pd.read_csv(ground_truth_poses_file_path)

    squared_deltas = (df[['x', 'y']] - df[['x', 'y']].shift(periods=1)) ** 2  # equivalent to (x_2-x_1)**2, (y_2-y_1)**2, for each row
    sum_of_squared_deltas = squared_deltas['x'] + squared_deltas['y']  # equivalent to (x_2-x_1)**2 + (y_2-y_1)**2, for each row
    euclidean_distance_of_deltas = np.sqrt(sum_of_squared_deltas)  # equivalent to sqrt( (x_2-x_1)**2 + (y_2-y_1)**2 ), for each row
    trajectory_length = euclidean_distance_of_deltas.sum()

    return float(trajectory_length)


def estimated_pose_trajectory_length_metric(estimated_poses_file_path):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("compute_trajectory_length: estimated_poses file not found {}".format(estimated_poses_file_path))
        return None

    df = pd.read_csv(estimated_poses_file_path)

    squared_deltas = (df[['x', 'y']] - df[['x', 'y']].shift(periods=1)) ** 2  # equivalent to (x_2-x_1)**2, (y_2-y_1)**2, for each row
    sum_of_squared_deltas = squared_deltas['x'] + squared_deltas['y']  # equivalent to (x_2-x_1)**2 + (y_2-y_1)**2, for each row
    euclidean_distance_of_deltas = np.sqrt(sum_of_squared_deltas)  # equivalent to sqrt( (x_2-x_1)**2 + (y_2-y_1)**2 ), for each row
    trajectory_length = euclidean_distance_of_deltas.sum()

    return float(trajectory_length)


# def test_metrics(run_output_folder):
#
#     run_info_path = path.join(run_output_folder, "run_info.yaml")
#     if not path.exists(run_info_path) or not path.isfile(run_info_path):
#         print_error("run info file does not exists")
#
#     with open(run_info_path) as run_info_file:
#         run_info = yaml.safe_load(run_info_file)
#
#     environment_folder = run_info['environment_folder']
#     metrics_result_folder_path = path.join(run_output_folder, "metric_results")
#     ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
#     ground_truth_map = GroundTruthMap(ground_truth_map_info_path)
#
#     # localization metrics
#     estimated_poses_path = path.join(run_output_folder, "benchmark_data", "estimated_poses.csv")
#     ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses.csv")
#
#     absolute_error_vs_geometric_similarity(estimated_poses_path, ground_truth_poses_path, ground_truth_map).to_csv(path.join(metrics_result_folder_path, "absolute_error_vs_geometric_similarity.csv"))
#
#
# if __name__ == '__main__':
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/elysium/performance_modelling/output/localization/*"))))
#     # for progress, run_folder in enumerate(run_folders):
#     #     print_info("main: compute_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
#     #     test_metrics(path.expanduser(run_folder))
#
#     run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_localization/run_000000000/"))))
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_slam/run_000000000/"))))
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/elysium/performance_modelling/output/localization/*"))))
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_localization/*"))))
#     last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
#     print("last run folder:", last_run_folder)
#     test_metrics(last_run_folder)
