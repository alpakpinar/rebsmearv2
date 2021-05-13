#!/usr/bin/env python

import os
import sys
import re
import math
import yaml
import argparse
import numpy as np
import uproot

from matplotlib import pyplot as plt
from rebsmearv2.helpers.paths import rebsmear_path
from pprint import pprint

pjoin = os.path.join

def load_xs():
    xsfile = rebsmear_path('data/xs.yml')
    with open(xsfile,'r') as f:
        xs_yml = yaml.load(f, Loader=yaml.FullLoader)

    xs = {dataset: _xs['gen'] for dataset, _xs in xs_yml.items()}

def lumi(year):
    """Golden JSON luminosity per for given year
    :param year: Year of data taking
    :type year: int
    :return: Golden JSON luminosity for that year in pb (!)
    :rtype: float
    """
    if year==2018:
        return 59.7
    if year==2017:
        return 41.5
    if year==2016:
        return 35.9

def merge_datasets():
    pass

if __name__ == '__main__':
    load_xs()