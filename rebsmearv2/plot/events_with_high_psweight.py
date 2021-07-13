#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

npy_binnings = {
    'mjj' : [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.],
    'ak4_pt0' : list(range(0,800,20)),
    'ak4_pt1' : list(range(0,800,20)),
    'ak4_eta0' : np.linspace(-5,5),
    'ak4_eta1' : np.linspace(-5,5),
    'ht' : np.arange(0,2000,50),
    'htmiss' : np.arange(0,400,20),
}

def plot_events_with_high_psweight(acc, outtag, distribution='mjj'):
    '''Plot distributions of the events with large prescale weights.'''
    treename = 'high_ps_events'
    acc.load(treename)
    t = acc[treename]

    vals = t[distribution].value
    fig, ax = plt.subplots()
    
    ax.hist(vals, bins=npy_binnings[distribution])
    outdir = f'./output/{outtag}/high_ps_weights'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{distribution}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0]

    distributions = [
        'mjj',
        'ak4_pt0',
        'ak4_pt1',
        'ak4_eta0',
        'ak4_eta1',
    ]
    
    for distribution in distributions:
        plot_events_with_high_psweight(acc, outtag, distribution=distribution)

if __name__ == '__main__':
    main() 
