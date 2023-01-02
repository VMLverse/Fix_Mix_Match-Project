#Utility Imports
import argparse
import yaml
import math
import os
import time
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='CS7643 Final Project')
parser.add_argument('--config')

#Setup Parser & Load yaml
global args
args = parser.parse_args()
print(args)
with open(args.config) as f:
    config = yaml.full_load(f)
#Load config_*.yaml into args
for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)
print("Config Parameters Loaded:", args)
