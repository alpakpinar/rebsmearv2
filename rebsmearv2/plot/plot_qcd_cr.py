#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from rebsmearv2.plot.util import scale_xs_lumi_sumw, rs_merge_datasets
from rebsmearv2.helpers.paths import rebsmear_path
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

PRETTY_LEGEND_LABELS = {
    '.*ZJetsToNuNu.*' : r'QCD $Z(\nu\nu)$',
    '.*WJetsToLNu.*' : r'QCD $W(\ell\nu)$',
    '.*DYJetsToLL.*' : r'QCD $Z(\ell\ell)$',
    'EWKZ2Jets.*ZToNuNu.*' : r'EWK $Z(\nu\nu)$',
    'EWKZ2Jets.*ZToLL.*' : r'EWK $Z(\ell\ell)$',
    'EWKW2Jets.*' : r'EWK $W(\ell\nu)$',
    'Top_FXFX.*' : 'Top',
    'Diboson.*' : 'Diboson',
    'MET_.*' : 'Data'
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath_vbf', help='Path to the coffea output from VBF processor.')
    parser.add_argument('inpath_rs', help='Path to the coffea output from rebalance and smear.')
    args = parser.parse_args()
    return args

def stack_plot_qcd_cr(acc, distribution, region='cr_vbf_qcd_rs', year=2017, rs_filepath=None):
    '''Calculate the data - (nonQCD MC) in the QCD CR.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution in BINNINGS.keys():
        new_ax = BINNINGS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('region', region)
    data = f'MET_{year}'
    mc = re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_M-50_HT_MLM.*|WJetsToLNu.*HT.*).*{year}')

    fig, ax, rax = fig_ratio()
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }
    
    # Observed data in QCD CR
    hist.plot1d(h[data], 
        ax=ax, 
        overlay='dataset', 
        binwnorm=1, 
        error_opts=data_err_opts
    )
    
    # Non QCD MC in QCD CR
    hist.plot1d(h[mc], 
        ax=ax, 
        overlay='dataset', 
        binwnorm=1, 
        stack=True,
        clear=False
    )

    if rs_filepath:
        h_rs = uproot.open(rs_filepath)[f'rebsmear_qcd_{year}_CR']    

    xedges = h.integrate('dataset').axes()[0].edges()
    xcenters = h.integrate('dataset').axes()[0].centers()

    hep.histplot(h_rs.values, h_rs.edges, ax=ax, binwnorm=1, label='R&S QCD Estimate')

    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e6)
    ax.set_ylabel('Events / GeV')

    ax.yaxis.set_ticks_position('both')

    ax.text(0,1,year,
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )
    
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        for regex, new_label in PRETTY_LEGEND_LABELS.items():
            if re.match(regex, label):
                handle.set_label(new_label)

    ax.legend(title='VBF QCD CR', handles=handles, ncol=2)

    # Calculate data / (non-QCD MC + R&S QCD estimate)
    sumw_data, sumw2_data = h[data].integrate('dataset').values(sumw2=True)[()]
    sumw_mc, sumw2_mc = h[mc].integrate('dataset').values(sumw2=True)[()]
    
    r = sumw_data / sumw_mc

    rerr = np.abs(
        hist.poisson_interval(r, sumw2_data / sumw_mc**2) - r
    ) 

    rax.errorbar(xcenters, r, yerr=rerr, marker='o', ls='', color='k')
    rax.grid(True)
    rax.set_ylabel('Data / Non-QCD MC')
    rax.set_ylim(0,2)

    rax.yaxis.set_ticks_position('both')

    unity = np.ones_like(sumw_mc)
    denom_unc = hist.poisson_interval(unity, sumw2_mc / sumw_mc ** 2)
    opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}
    
    rax.fill_between(
        xedges,
        np.r_[denom_unc[0], denom_unc[0, -1]],
        np.r_[denom_unc[1], denom_unc[1, -1]],
        **opts
    )

    outdir = './output/qcd_cr'
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    outpath = pjoin(outdir, f'qcd_cr_{distribution}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath_vbf = rebsmear_path('submission/vbfhinv/merged_2021-07-12_vbfhinv_ULv8_05Feb21_rebsmearCR')

    rs_filepath = rebsmear_path('plot/output/merged_2021-07-05_rebsmear_v2_run_PFJet40_suppress/rebsmear_qcd_estimate.root')

    acc_vbf = dir_archive(inpath_vbf)
    acc_vbf.load('sumw')
    acc_vbf.load('sumw_pileup')
    acc_vbf.load('nevents')

    stack_plot_qcd_cr(acc_vbf, distribution='mjj', rs_filepath=rs_filepath)

if __name__ == '__main__':
    main()