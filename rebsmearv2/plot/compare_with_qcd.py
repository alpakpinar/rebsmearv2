#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import mplhep as hep

from rebsmearv2.helpers.paths import rebsmear_path
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

def main():
    # Compare the QCD template from R&S and IC
    rspath = rebsmear_path('plot/output/merged_2021-07-05_rebsmear_v2_run_PFJet40_suppress/rebsmear_qcd_estimate.root')
    rebsmear_file = uproot.open(rspath)

    qcd_file = uproot.open(
        rebsmear_path('input/qcd_from_ic/out_MTR_2017.root_qcdDD.root')
    )

    outdir = os.path.dirname(rspath)

    h_qcd = qcd_file['rebin_QCD_hist_counts']
    h_rebsmear = rebsmear_file['rebsmear_qcd_2017']

    fig, ax = plt.subplots()
    hep.histplot(h_qcd.values, h_qcd.edges, yerr=np.sqrt(h_qcd.variances), binwnorm=1, label='QCD (IC)')
    hep.histplot(h_rebsmear.values, h_rebsmear.edges, yerr=np.sqrt(h_rebsmear.variances), binwnorm=1, label='Rebalance & Smear')

    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_ylabel('Events / GeV')
    ax.legend()

    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e2)

    outpath = pjoin(outdir, 'qcd_comparison.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

if __name__ == '__main__':
    main()