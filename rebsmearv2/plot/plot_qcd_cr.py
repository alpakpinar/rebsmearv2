#!/usr/bin/env python

import os
import sys
import re
import argparse
from typing import BinaryIO
import numpy as np

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

def extract_yields_in_cr(acc, distribution, region='cr_vbf_qcd_rs', year=2017):
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
    
    hist.plot1d(h[data], 
        ax=ax, 
        overlay='dataset', 
        binwnorm=1, 
        error_opts=data_err_opts
    )
    
    hist.plot1d(h[mc], 
        ax=ax, 
        overlay='dataset', 
        binwnorm=1, 
        stack=True,
        clear=False
    )

    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e6)
    ax.set_ylabel('Events / GeV')

    ax.yaxis.set_ticks_position('both')

    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        for regex, new_label in PRETTY_LEGEND_LABELS.items():
            if re.match(regex, label):
                handle.set_label(new_label)

    ax.legend(title='VBF QCD CR', handles=handles, ncol=2)

    # Calculate data - MC
    h_data = h[data].integrate('dataset')
    h_mc = h[mc].integrate('dataset')
    h_mc.scale(-1)
    h_data.add(h_mc)

    # Plot data - MC on the bottom pad
    hist.plot1d(h_data, ax=rax, binwnorm=1)

    rax.set_ylabel('(Data - MC) / GeV')
    rax.set_ylim(1e-3,1e1)
    rax.set_yscale('log')

    rax.get_legend().remove()

    rax.yaxis.set_ticks_position('both')

    outdir = './output/qcd_cr'
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    outpath = pjoin(outdir, f'qcd_cr_{distribution}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

    # Return the QCD yield
    return h_data

def plot_rebsmear_prediction(acc_rs, h_qcd, distribution='mjj', dataset='JetHT_2017', region='cr_vbf_qcd'):
    '''Together with the data - MC prediction from VBF, plot the rebalance and smear prediction.'''
    acc_rs.load(distribution)
    h = acc_rs[distribution]

    # Merge the JetHT datasets together
    h = rs_merge_datasets(h)

    if distribution in BINNINGS.keys():
        new_ax = BINNINGS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('region', region)[dataset]

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overlay='dataset', binwnorm=1)
    hist.plot1d(h_qcd, ax=ax, binwnorm=1, clear=False)

    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e2)

    fig.savefig('test.pdf')

def main():
    inpath_vbf = rebsmear_path('submission/vbfhinv/merged_2021-06-11_vbfhinv_ULv8_05Feb21_rebsmear_CR')
    inpath_rs = rebsmear_path('submission/merged_2021-06-11_rebsmear_privatePS')

    acc_vbf = dir_archive(inpath_vbf)
    acc_vbf.load('sumw')
    acc_vbf.load('sumw_pileup')
    acc_vbf.load('nevents')

    h_qcd = extract_yields_in_cr(acc_vbf, distribution='mjj')

    # Rebalance and smear output
    acc_rs = dir_archive(inpath_rs)

    plot_rebsmear_prediction(acc_rs, h_qcd)

if __name__ == '__main__':
    main()