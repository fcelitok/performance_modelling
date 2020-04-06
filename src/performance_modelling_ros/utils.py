#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
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


def print_info(*args, **kwargs):
    if 'replace_previous_line' in kwargs and kwargs['replace_previous_line']:
        sys.stdout.write("\033[F\033[K"*kwargs['replace_previous_line'])
    print(colored(' '.join(map(str, args)), 'blue', attrs=['bold']))


def print_error(*args, **kwargs):
    if 'replace_previous_line' in kwargs and kwargs['replace_previous_line']:
        sys.stdout.write("\033[F\033[K"*kwargs['replace_previous_line'])
    print(colored(' '.join(map(str, args)), 'red', attrs=['bold']))


# noinspection PyBroadException
def print_fatal(*args):
    text = ' '.join(map(str, args))
    try:
        text_lines = text.split('\n')
        print(text_lines)
        n = max(map(len, text_lines))
        print(n)
        colored_text = colored(text, 'red', attrs=['bold', 'blink'])
        b = colored('*', 'red', attrs=['bold', 'blink', 'reverse'])
        print("{h_border}\n"
              "{v_border} {spaces} {v_border}".format(h_border=b*(n+4), v_border=b, spaces=' '*n))
        for line in text_lines:
            colored_line = colored(line.ljust(n, ' '), 'red', attrs=['bold'])
            print("{v_border} {line} {v_border}".format(v_border=b, line=colored_line))
        print("{v_border} {spaces} {v_border}\n"
              "{h_border}".format(h_border=b*(n+4), v_border=b, spaces=' '*n, text=colored_text))
    except:
        print(text)
