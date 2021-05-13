#!/usr/bin/env python

import os
import sys
import re
import math
import argparse
import numpy as np
import uproot

from matplotlib import pyplot as plt
from rebsmearv2.helpers.paths import rebsmear_path
from pprint import pprint

pjoin = os.path.join

def main():
    # Path to the directory containing list of ROOT input files (R&S trees)
    inpath = sys.argv[1]
    trees = [pjoin(inpath, t) for t in os.listdir(inpath) if re.match('.*tree_(\d+).root', t)]

if __name__ == '__main__':
    main()