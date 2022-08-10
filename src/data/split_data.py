#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script divide train, validation and testing data"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import argparse
import os

# Third-party modules
import splitfolders
import yaml

def split_data(data_dir, ratio):
    # Split data into train, val and test folders
    splitfolders.ratio(data_dir, output=data_dir, seed=1337, ratio=ratio)

if __name__ == "__main__":
    # Define data directory
    data_dir = os.path.join('..','..', 'data', 'preprocessed')
    # Initialize Argument Parser
    parser = argparse.ArgumentParser()
    # Add arguments we want to parse
    parser.add_argument("--ratio", nargs="+", type=float)
    # Read arguments from command line
    args = parser.parse_args()
    ratio = tuple(args.ratio)
    # Run function
    split_data(data_dir, ratio)