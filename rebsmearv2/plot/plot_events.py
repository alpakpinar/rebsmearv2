#!/usr/bin/env python

import os
import sys
import re
import math
import argparse
import numpy as np

from matplotlib import pyplot as plt
from coffea import hist
from rebsmearv2.plot.util import scale_xs_lumi_sumw, merge_datasets
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

BINNINGS = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.]),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(0,800,20)) ),
    'ak4_pt1' : hist.Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(0,800,20)) ),
    'ak4_eta0' : hist.Bin('jeteta',r'Leading AK4 jet $\eta$', 50, -5, 5),
    'ak4_eta1' : hist.Bin('jeteta',r'Trailing AK4 jet $\eta$', 50, -5, 5),
    'ak4_phi0' : hist.Bin('jetphi',r'Leading AK4 jet $\phi$', 50, -np.pi, np.pi),
    'ak4_phi1' : hist.Bin('jetphi',r'Trailing AK4 jet $\phi$', 50, -np.pi, np.pi),
    'ht' : hist.Bin("ht", r"$H_{T}$ (GeV)", 20, 0, 4000),
    'htmiss' : hist.Bin("ht", r"$H_{T}^{miss}$ (GeV)", 20, 0, 4000),
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to merged coffea files.')
    parser.add_argument('--years', nargs='*', type=int, default=[2017,2018], help='Years to run on, default is both 17 and 18.')
    parser.add_argument('--region', default='.*', help='Regions to run on.')
    args = parser.parse_args()
    return args

def make_plot(acc, distribution, outdir='./output', region='sr_vbf', dataset='QCD_HT', years=[2017,2018]):
    acc.load(distribution)
    h = acc[distribution]

    scale_xs_lumi_sumw(h, acc)
    h = merge_datasets(h)

    # Rebinning
    try:
        new_ax = BINNINGS[distribution]
        h = h.rebin(new_ax.name, new_ax)
    except KeyError:
        pass

    h = h.integrate('region', region)

    for year in years:
        fig, ax = plt.subplots()
        _h = h[re.compile(f'{dataset}.*{year}')]
        hist.plot1d(_h, ax=ax, binwnorm=1, overlay='dataset')
    
        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e10)
    
        ax.text(0.,1.,year,
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'{dataset}_{year}_{region}_{distribution}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    args = parse_cli()
    # Path to the directory containing list of ROOT input files (R&S trees)
    inpath = args.inpath
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    try:
        outtag = re.findall('merged_.*', inpath)[0]
    except KeyError:
        raise RuntimeError(f'Check the naming of input: {os.path.basename(inpath)}')

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    distributions = BINNINGS.keys()

    regions = [
        'inclusive',
        'sr_vbf'
    ]
    
    for region in regions:
        if not re.match(args.region, region):
            continue
        for distribution in distributions:
            make_plot(acc, 
                outdir=outdir, 
                distribution=distribution,
                region=region,
                years=args.years
            )

if __name__ == '__main__':
    main()