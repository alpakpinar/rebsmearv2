#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath1', help='Path to ROOT file containing first template')
    parser.add_argument('inpath2', help='Path to ROOT file containing second template')
    parser.add_argument('--signal', action='store_true', help='Compare templates for the signal region (otherwise, QCD CR)')
    args = parser.parse_args()
    return args

def main():
    args = parse_cli()
    f1 = uproot.open(args.inpath1)
    f2 = uproot.open(args.inpath2)

    histname = 'rebsmear_qcd_2017' if args.signal else 'rebsmear_qcd_2017_CR'

    h1 = f1[histname]
    h2 = f2[histname]

    fig, ax = plt.subplots()
    hep.histplot(h1.values, h1.edges, ax=ax, binwnorm=1, label=args.inpath1.split('/')[-2])
    hep.histplot(h2.values, h2.edges, ax=ax, binwnorm=1, label=args.inpath2.split('/')[-2])

    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e6)
    ax.set_ylabel('Events / GeV')
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')

    ax.legend()

    figtext = '2017 SR' if args.signal else '2017 CR'
    ax.text(0,1,figtext,
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    outdir = './template_comparison'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{"sr" if args.signal else "cr"}_comparison.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    r = h2.values / h1.values
    print(r)

if __name__ == '__main__':
    main()