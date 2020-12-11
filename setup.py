#!/usr/bin/env python

from distutils.core import setup

setup(
    name="performance-modelling-py",
    version="0.0.5",
    author="Enrico Piazza",
    author_email="erico.piazza@polimi.it",
    description="TODO",
    packages=['performance_modelling_py'],
    package_dir={'performance_modelling_py': './performance_modelling_py'},
    scripts=[
        'performance_modelling_py/data_manipulation/collect_run_results.py',
        'performance_modelling_py/environment/compress_dataset_files_and_remove.py',
        'performance_modelling_py/environment/compress_dataset_files.py',
        'performance_modelling_py/environment/decompress_dataset_files.py',
        'performance_modelling_py/visualisation/save_maps_from_bag.py',
    ],
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # packages=setuptools.find_packages(),
    # classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    # ],
    # python_requires='>=3.6',
)
