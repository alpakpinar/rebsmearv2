#!/usr/bin/env python

import os
import sys
import re
import math
import argparse
import uproot
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from rebsmearv2.plot.util import scale_xs_lumi_sumw, rs_merge_datasets, URTH1
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

Bin = hist.Bin

BINNINGS = {
    'mjj' : Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.]),
    'ak4_pt0' : Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(0,800,40)) ),
    'ak4_pt1' : Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(0,800,40)) ),
    'ak4_eta0' : Bin('jeteta',r'Leading AK4 jet $\eta$', 25, -5, 5),
    'ak4_eta1' : Bin('jeteta',r'Trailing AK4 jet $\eta$', 25, -5, 5),
    'ak4_phi0' : Bin('jetphi',r'Leading AK4 jet $\phi$', 25, -np.pi, np.pi),
    'ak4_phi1' : Bin('jetphi',r'Trailing AK4 jet $\phi$', 25, -np.pi, np.pi),
    'ht' : Bin("ht", r"$H_{T}$ (GeV)", 50, 0, 4000),
    'htmiss' : Bin("ht", r"$H_{T}^{miss}$ (GeV)", 80, 0, 800),
    'ak4_nef0' : Bin("frac", r"Leading Jet Neutral EM Fraction", 50, 0, 1),
    'ak4_nhf0' : Bin("frac", r"Leading Jet Neutral Hadron Fraction", 50, 0, 1),
    'ak4_cef0' : Bin("frac", r"Leading Jet Charged EM Fraction", 50, 0, 1),
    'ak4_chf0' : Bin("frac", r"Leading Jet Charged Hadron Fraction", 50, 0, 1),
    'ak4_nef1' : Bin("frac", r"Trailing Jet Neutral EM Fraction", 50, 0, 1),
    'ak4_nhf1' : Bin("frac", r"Trailing Jet Neutral Hadron Fraction", 50, 0, 1),
    'ak4_cef1' : Bin("frac", r"Trailing Jet Charged EM Fraction", 50, 0, 1),
    'ak4_chf1' : Bin("frac", r"Trailing Jet Charged Hadron Fraction", 50, 0, 1),
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to merged coffea files.')
    parser.add_argument('--years', nargs='*', type=int, default=[2017,2018], help='Years to run on, default is both 17 and 18.')
    parser.add_argument('--distribution', default='.*', help='Distribution to plot.')
    parser.add_argument('--region', default='.*', help='Regions to run on.')
    args = parser.parse_args()
    return args

def make_plot(acc, distribution, outdir='./output', region='sr_vbf', dataset='QCD_HT', years=[2017,2018], outrootfile=None):
    acc.load(distribution)
    h = acc[distribution]

    scale_xs_lumi_sumw(h, acc)
    h = rs_merge_datasets(h)

    # Rebinning
    try:
        new_ax = BINNINGS[distribution]
        h = h.rebin(new_ax.name, new_ax)
    except KeyError:
        pass

    h = h.integrate('region', region)

    for year in years:
        fig, ax = plt.subplots()
        _h = h[re.compile(f'{dataset}.*{year}')].integrate('dataset')
        
        sumw, sumw2 = _h.values(sumw2=True)[()]
        xedges = _h.axes()[0].edges()

        # Correct for sumw2 with all the toys
        Ntoys = 1e3
        yerr=np.sqrt(sumw2*Ntoys)
        hep.histplot(sumw, xedges, yerr=yerr, binwnorm=1, label=f'{dataset} {year}')

        ax.legend(title='Dataset')
        ax.set_yscale('log')
        ax.set_ylim(1e-4,1e8)
        ax.set_ylabel('Events / GeV')
        ax.set_xlabel(_h.axes()[0].label)
        ax.yaxis.set_ticks_position('both')

        ax.text(0.,1.,year,
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        # Calculate % of events with HTmiss > 250 GeV
        if distribution == 'htmiss':
            centers = h.axis('ht').centers()
            sumw_all = np.sum(_h.values()[()]) 
            sumw_high = np.sum(_h.values()[()][centers>250.])

            ax.text(1., 1., f'$H_T^{{miss}} > 250 \\ GeV: {sumw_high/sumw_all * 100:.3f}\\%$',
                fontsize=14,
                ha='right',
                va='bottom',
                transform=ax.transAxes
            )

        if distribution == 'mjj' and region == 'sr_vbf':
            # Underflow + overflow bins = 0
            sumw = np.r_[0, sumw, 0]
            sumw2 = np.r_[0, sumw2*Ntoys, 0]
            outrootfile[f'rebsmear_qcd_{year}'] = URTH1(edges=xedges, sumw=sumw, sumw2=sumw2)
        elif distribution == 'mjj' and region == 'cr_vbf_qcd':
            # Underflow + overflow bins = 0
            sumw = np.r_[0, sumw, 0]
            sumw2 = np.r_[0, sumw2*Ntoys, 0]
            outrootfile[f'rebsmear_qcd_{year}_CR'] = URTH1(edges=xedges, sumw=sumw, sumw2=sumw2)

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
        'sr_vbf',
        'cr_vbf_qcd'
    ]
    
    outrootfile = uproot.recreate(pjoin(outdir, 'rebsmear_qcd_estimate.root'))

    for region in regions:
        if not re.match(args.region, region):
            continue
        for distribution in distributions:
            if not re.match(args.distribution, distribution):
                continue
            
            make_plot(acc, 
                outdir=outdir, 
                distribution=distribution,
                region=region,
                dataset='JetHT',
                years=args.years,
                outrootfile=outrootfile
            )

if __name__ == '__main__':
    main()
