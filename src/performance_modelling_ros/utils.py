#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from os import path
try:
    from termcolor import colored
except ImportError as e:
    print("\n"
          "\033[33mCould not import termcolor.\n"
          "Install termcolor by running\n"
          "    sudo apt install python-termcolor\033[0m\n")

    def colored(text, *args, **kwargs):
        return text


def backup_file_if_exists(target_path):
    if path.exists(target_path):
        backup_path = path.abspath(target_path) + '.backup'
        backup_file_if_exists(backup_path)
        print("backup_file_if_exists: {} -> {}".format(target_path, backup_path))
        os.rename(target_path, backup_path)


def print_info(*args):
    print(colored(' '.join(map(str, args)), 'blue', attrs=['bold']))


def print_error(*args):
    print(colored(' '.join(map(str, args)), 'red', attrs=['bold']))


def print_fatal(*args):
    text = ' '.join(map(str, args))
    colored_text = colored(text, 'red', attrs=['bold', 'blink'])
    b = colored('*', 'red', attrs=['bold', 'blink', 'reverse'])
    n = len(text)
    print(n)
    print("{h_border}\n"
          "{v_border} {spaces} {v_border}\n"
          "{v_border} {text} {v_border}\n"
          "{v_border} {spaces} {v_border}\n"
          "{h_border}".format(h_border=b*(n+4), v_border=b, spaces=' '*n, text=colored_text))
