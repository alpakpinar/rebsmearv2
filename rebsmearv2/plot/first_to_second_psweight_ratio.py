#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from matplotlib import pyplot as plt

pjoin = os.path.join

def main():
    # Path to the combined ROOT file containing rebalanced events
    inpath = sys.argv[1]
    t = uproot.open(inpath)['Events']

    ratio = t['second_to_first_prescale_ratio'].array()

    fig, ax = plt.subplots()

    ratio_bins = np.logspace(0,6)

    ax.hist(ratio, bins=ratio_bins)

    ax.set_xlabel('Second PS / PS')
    ax.set_xscale('log')
    ax.set_xlim(1e0,1e6)

    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e8)
    ax.set_title('JetHT_2017')
    fig.savefig('./output/psweight_ratio.pdf')

if __name__ == '__main__':
    main()