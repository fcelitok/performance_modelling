#!/usr/bin/env python

from __future__ import print_function

import argparse
import math
import numpy as np
import random
import os
import sys
from os import path
from scipy.stats import t
from subprocess import Popen

import rosbag
import tf
import tf.transformations

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import savefig


def write_ground_truth(bag_file_path, output_file_path):
    """
    Given a bag file writes out a file with the poses from topic /base_pose_ground_truth
    """

    ground_truth = open(output_file_path, "w")

    first_run = True
    displacement_x = 0
    displacement_y = 0

    for topic, msg, _ in rosbag.Bag(bag_file_path).read_messages():
        if topic == "/base_pose_ground_truth":
            position = msg.pose.pose.position
            quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
            _, _, theta = tf.transformations.euler_from_quaternion(quaternion)

            if first_run:
                displacement_x = -position.x
                displacement_y = -position.y
                first_run = False

            ground_truth.write("{t} {x} {y} {theta}\n".format(t=msg.header.stamp.to_sec(), x=position.x + displacement_x, y=position.y + displacement_y, theta=theta))

            # What the f**k was this?
            # if str(msg.header.stamp)[:-9] == "":
            #     ground_truth.write("0." + str(msg.header.stamp)[-9:] + " " + str(position.x + displacement_x) + " " + str(
            #         position.y + displacement_y) + " " + str(rot[2]) + "\n")
            # else:
            #     ground_truth.write(str(msg.header.stamp)[:-9] + "." + str(msg.header.stamp)[-9:] + " " + str(
            #         position.x + displacement_x) + " " + str(position.y + displacement_y) + " " + str(rot[2]) + "\n")

    ground_truth.close()


def generate_relations_o_and_re(run_output_folder, base_link_poses_file_path, ground_truth_file_path, metric_evaluator_exec_path, seconds=0.5, alpha=0.99, max_error=0.02, skip_ordered_recomputation=False):
    """
    Generates the Ordered and the Random relations files
    """
    ground_truth_dict = dict()

    if seconds == '0':
        print("generate_relations_o_and_re: argument seconds can not be 0")
        return

    ground_truth_file = open(ground_truth_file_path, "r")

    # builds dictionary with all the ground truth
    for line in ground_truth_file:
        time, x, y, theta = line.split(" ")
        ground_truth_dict[float(time)] = [float(x), float(y), float(theta)]

    ground_truth_file.close()

    if not path.exists(path.join(run_output_folder, "Relations/Original/")):
        os.makedirs(path.join(run_output_folder, "Relations/Original/"))

    # RE
    relations_re_file_path = path.join(run_output_folder, "Relations/Original/", "RE.relations")
    relations_file_re = open(relations_re_file_path, "w")

    n_samples = 500
    i = 0

    while i < n_samples:
        first_stamp = float(random.choice(ground_truth_dict.keys()))
        second_stamp = float(random.choice(ground_truth_dict.keys()))
        if first_stamp > second_stamp:
            temp = first_stamp
            first_stamp = second_stamp
            second_stamp = temp
        first_pos = ground_truth_dict[first_stamp]
        second_pos = ground_truth_dict[second_stamp]

        rel = get_matrix_diff(first_pos, second_pos)

        x = rel[0, 3]
        y = rel[1, 3]
        theta = math.atan2(rel[1, 0], rel[0, 0])

        relations_file_re.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

        i += 1

    relations_file_re.close()

    # now we invoke the metric evaluator on this relations file, we read the sample standard
    # deviation and we exploit it to rebuild a better sample

    # compute translational sample size
    summary_t_file_path = path.join(run_output_folder, "summaryT.error")
    p1 = Popen([metric_evaluator_exec_path, "-s", base_link_poses_file_path, "-r", relations_re_file_path, "-w", "{1.0,1.0,1.0,0.0,0.0,0.0}", "-e", summary_t_file_path])
    p1.wait()
    error_file = open(summary_t_file_path, "r")
    content = error_file.readlines()
    words = content[1].split(", ")
    std = float(words[1])
    var = math.pow(std, 2)
    z_a_2 = t.ppf(alpha, n_samples - 1)
    delta = max_error
    n_samples_t = math.pow(z_a_2, 2) * var / math.pow(delta, 2)

    # compute rotational sample size
    summary_r_file_path = path.join(run_output_folder, "summaryR.error")
    p1 = Popen([metric_evaluator_exec_path, "-s", base_link_poses_file_path, "-r", relations_re_file_path, "-w", "{0.0,0.0,0.0,1.0,1.0,1.0}", "-e", summary_r_file_path])
    p1.wait()
    error_file = open(summary_r_file_path, "r")
    content = error_file.readlines()
    words = content[1].split(", ")
    std = float(words[1])
    var = math.pow(std, 2)
    z_a_2 = t.ppf(alpha, n_samples - 1)
    delta = max_error
    n_samples_r = math.pow(z_a_2, 2) * var / math.pow(delta, 2)

    # select the biggest of the two
    n_samples = max(n_samples_t, n_samples_r)
    print(n_samples_t)
    print(n_samples_r)
    print(n_samples)

    relations_file_re = open(relations_re_file_path, "w")

    i = 0
    while i < n_samples:
        first_stamp = float(random.choice(ground_truth_dict.keys()))
        second_stamp = float(random.choice(ground_truth_dict.keys()))
        if first_stamp > second_stamp:
            temp = first_stamp
            first_stamp = second_stamp
            second_stamp = temp
        first_pos = ground_truth_dict[first_stamp]
        second_pos = ground_truth_dict[second_stamp]

        rel = get_matrix_diff(first_pos, second_pos)

        x = rel[0, 3]
        y = rel[1, 3]
        theta = math.atan2(rel[1, 0], rel[0, 0])

        relations_file_re.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

        i += 1

    relations_file_re.close()

    if not path.exists(path.join(run_output_folder, "Errors/Original/RE/")):
        os.makedirs(path.join(run_output_folder, "Errors/Original/RE/"))

    metric_evaluator_t_errors_path = path.join(run_output_folder, "Errors/Original/RE/T.errors")
    metric_evaluator_t_unsorted_errors_path = path.join(run_output_folder, "Errors/Original/RE/T-unsorted.errors")
    p2 = Popen([metric_evaluator_exec_path, "-s", base_link_poses_file_path, "-r", relations_re_file_path, "-w", "{1.0,1.0,1.0,0.0,0.0,0.0}", "-e", metric_evaluator_t_errors_path, "-eu", metric_evaluator_t_unsorted_errors_path])
    p2.wait()

    metric_evaluator_r_errors_path = path.join(run_output_folder, "Errors/Original/RE/R.errors")
    metric_evaluator_r_unsorted_output_path = path.join(run_output_folder, "Errors/Original/RE/R-unsorted.errors")
    p3 = Popen([metric_evaluator_exec_path, "-s", base_link_poses_file_path, "-r", relations_re_file_path, "-w", "{0.0,0.0,0.0,1.0,1.0,1.0}", "-e", metric_evaluator_r_errors_path, "-eu", metric_evaluator_r_unsorted_output_path])
    p3.wait()

    # ORDERED
    if not skip_ordered_recomputation:
        ordered_relations_file_path = path.join(run_output_folder, "Relations/Original/", "Ordered.relations")
        relations_file_ordered = open(ordered_relations_file_path, "w")

        ground_truth_sorted_indices = sorted(ground_truth_dict)

        idx = 1
        idx_delta = 10
        first_stamp = ground_truth_sorted_indices[idx]
        while idx + idx_delta < len(ground_truth_sorted_indices):
            second_stamp = ground_truth_sorted_indices[idx + idx_delta]
            if first_stamp in ground_truth_dict.keys():
                first_pos = ground_truth_dict[first_stamp]
                if second_stamp in ground_truth_dict.keys():
                    second_pos = ground_truth_dict[second_stamp]
                    rel = get_matrix_diff(first_pos, second_pos)

                    x = rel[0, 3]
                    y = rel[1, 3]
                    theta = math.atan2(rel[1, 0], rel[0, 0])

                    relations_file_ordered.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

            first_stamp = second_stamp
            idx += idx_delta

        relations_file_ordered.close()

        if not path.exists(path.join(run_output_folder, "Errors/Original/Ordered/")):
            os.makedirs(path.join(run_output_folder, "Errors/Original/Ordered/"))

        metric_evaluator_t_errors_path = path.join(run_output_folder, "Errors/Original/Ordered/T.errors")
        metric_evaluator_t_unsorted_errors_path = path.join(run_output_folder, "Errors/Original/Ordered/T-unsorted.errors")
        p4 = Popen([metric_evaluator_exec_path, "-s", base_link_poses_file_path, "-r", ordered_relations_file_path, "-w", "{1.0,1.0,1.0,0.0,0.0,0.0}", "-e", metric_evaluator_t_errors_path, "-eu", metric_evaluator_t_unsorted_errors_path])
        p4.wait()

        metric_evaluator_r_errors_path = path.join(run_output_folder, "Errors/Original/Ordered/R.errors")
        metric_evaluator_r_unsorted_errors_path = path.join(run_output_folder, "Errors/Original/Ordered/R-unsorted.errors")
        p5 = Popen([metric_evaluator_exec_path, "-s", base_link_poses_file_path, "-r", ordered_relations_file_path, "-w", "{0.0,0.0,0.0,1.0,1.0,1.0}", "-e", metric_evaluator_r_errors_path, "-eu", metric_evaluator_r_unsorted_errors_path])
        p5.wait()


def get_matrix_diff(p1, p2):
    """
    Computes the rototranslation difference of two points
    """

    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    theta1 = p1[2]
    theta2 = p2[2]

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


def plot_trajectory(traj_file_path):
    file2 = open(traj_file_path, 'r')

    x = []
    y = []

    for line in file2:
        words = line.split(" ")
        x.append(words[5])
        y.append(words[6])

    file2.close()
    plt.plot(x, y, 'r')


def plot_ground_truth_traj(traj_file_path):
    file2 = open(traj_file_path, 'r')

    x_gt = []
    y_gt = []

    for line in file2:
        words = line.split(" ")
        x_gt.append(words[1])
        y_gt.append(words[2])

    file2.close()
    plt.plot(x_gt, y_gt, 'g')


def save_plot(slam, gt, save):
    plot_trajectory(slam)
    savefig(save + "_slam.png")
    plt.clf()

    plot_ground_truth_traj(gt)
    savefig(save + "_gt.png")

    plot_trajectory(slam)
    savefig(save)


def save_plot2(base_link_poses_path, ground_truth_poses_path, figure_output_path):
    """
    Creates a figure with the trajectories slam and ground truth
    """
    base_link_poses = open(base_link_poses_path, 'r')

    x = []
    y = []

    for line in base_link_poses:
        words = line.split(" ")
        x.append(float(words[5]))
        y.append(float(words[6]))

    base_link_poses.close()

    ground_truth_poses = open(ground_truth_poses_path, 'r')

    x_gt = []
    y_gt = []

    for line in ground_truth_poses:
        words = line.split(" ")
        x_gt.append(float(words[1]))
        y_gt.append(float(words[2]))

    ground_truth_poses.close()

    fig, ax = plt.subplots()
    ax.cla()
    ax.plot(x, y, 'r', x_gt, y_gt, 'b')
    fig.savefig(figure_output_path)
    # plt.close(fig)
    plt.show()


def generate_all(run_output_folder, metric_evaluator_exec_path, skip_ground_truth_conversion=False, skip_ordered_recomputation=False):
    """
    Given a folder path if there is a .bag file and an Out.log file generates Relations, errors and trajectories
    """
    bag_file_path = path.join(run_output_folder, "odom_tf_ground_truth.bag")
    base_link_poses_path = path.join(run_output_folder, "base_link_poses")
    ground_truth_poses_path = path.join(run_output_folder, "ground_truth_poses")

    if not skip_ground_truth_conversion:
        if path.isfile(bag_file_path):
            write_ground_truth(bag_file_path, ground_truth_poses_path)
        else:
            print("generate_all: bag file not found {}".format(bag_file_path))

    if path.isfile(base_link_poses_path):
        generate_relations_o_and_re(run_output_folder, base_link_poses_path, ground_truth_poses_path, metric_evaluator_exec_path, skip_ordered_recomputation=skip_ordered_recomputation)
        save_plot2(base_link_poses_path, ground_truth_poses_path, path.join(run_output_folder, "trajectories.png"))
    else:
        print("generate_all: base_link_transforms file not found {}".format(base_link_poses_path))

    # writeText(run_output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Given the directory of an individual output run, this tool generates all the support files that are necessary for the RPPF to actually train the models and perform predictions. More specifically:\n\n\t1. If the second (optional) parameter is set to True, it converts the .bag ground truth trajectory data into a .log ground truth trajectory data. Otherwise, the .log file is assumed to be already present and the conversion is skipped;\n\t2. It creates a Relations folder and generates both the ordered and randomly sampled relations files;\n\t3. It invokes the Freiburg Metric Evaluator tool on the generated relations, storing the corresponding error files in an Error folder;\n\t4. It plots a trajectories.png file overlaying the ground truth trajectory (in green) with the estimated SLAM trajectory (in red);\n\t5. It uses the last available map snapshot and the freshly computed mean translation error from the randomly sampled relations to generate an overlayed errorMap.png file.\n\nUnder normal conditions, there is no need to manually execute this component, as it is automatically invoked by launch.py at the end of each exploration run. However, it can also be executed manually, for instance to compute the relations associated with existing real world datasets (e.g. the RAWSEEDS Bicocca indoor datasets).')
    parser.add_argument('folder_of_individual_output_run',
                        help='the folder that contains the output exploration data of an individual run')
    parser.add_argument('-s', '--skip_ground_truth_conversion', action='store_true',
                        help='skips the conversion of the ground truth data from the bag file to the log file and forces the re-usage of the existing ground truth log file; if the ground truth log file does not exist, the program will crash')
    args = parser.parse_args()

    # Find the metric_evaluator executable
    import roslib.packages
    metric_evaluator_package_name = 'performance_modelling'
    metric_evaluator_exec_name = 'metricEvaluator'
    metric_evaluator_resources_list = roslib.packages.find_resource(metric_evaluator_package_name, metric_evaluator_exec_name)
    if len(metric_evaluator_resources_list) > 1:
        print("Multiple files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        sys.exit(-1)
    elif len(metric_evaluator_resources_list) == 0:
        print("No files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        sys.exit(-1)
    _metric_evaluator_exec_path = metric_evaluator_resources_list[0]

    generate_all(args.folder_of_individual_output_run, metric_evaluator_exec_path=_metric_evaluator_exec_path, skip_ground_truth_conversion=args.skip_ground_truth_conversion)  # TODO update arguments or get rid of this
