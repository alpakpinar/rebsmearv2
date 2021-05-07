#!/usr/bin/env python

import os
import sys
import re
import glob
import uproot
import ROOT as r
import mplhep as hep
from matplotlib import pyplot as plt
from coffea.util import load
from coffea import hist
from pprint import pprint

pjoin = os.path.join

def htmiss_func_name():
    return 'gen_htmiss_pt'

def tag_to_plottag(dataset_tag):
    mapping = {
        'jetht' : 'JetHT',
        'qcd' : 'QCD MC',
    }
    return mapping[dataset_tag]

def save_htmiss_before_and_after(infiles, outdir):
    '''Save the HTmiss histograms before and after rebalancing, using the list of provided input files.'''
    # Initialize two ROOT histograms (before/after)
    h_bef = r.TH1F('htmiss_before', r'$H_T^{miss} \ (GeV)$', 50, 0, 500)
    h_aft = r.TH1F('htmiss_after', r'$H_T^{miss} \ (GeV)$', 50, 0, 500)

    for infile in infiles:
        print(f'Reading events from file: {infile}')
        f = r.TFile(infile)

        keys = f.GetListOfKeys()
        nkeys = len(keys)

        for idx in range(nkeys):
            totalnumkeys = nkeys-1
            if idx % 100 == 0 and idx > 0:
                print(f'Reading entry: {idx}/{totalnumkeys} ({idx / totalnumkeys * 100:.2f}%)', end='\r')
            
            name = keys[idx].GetName()
            if name.startswith('ProcessID'):
                continue

            ws = f.Get(name)
            # Extract the HTmiss value  out of the workspace
            htmiss = ws.function(htmiss_func_name()).getValV()

            if name.startswith('before'):
                h_bef.Fill(htmiss)
            elif name.startswith('rebalanced'):
                h_aft.Fill(htmiss)

    # Once we're done filling the histogram with all the events, save the two histograms to a new ROOT file
    outpath = pjoin(outdir, 'htmiss_after_reb.root')
    outf = r.TFile(outpath, 'RECREATE')

    outf.cd()
    h_bef.Write()
    h_aft.Write()

    print(f'Histograms saved to: {outpath}')
    return outpath

def plot_htmiss_before_and_after(outdir, infile, dataset_tag='jetht', plot_gen=True):
    '''Do the actual plotting of distributions.'''
    f = uproot.open(infile)
    htmiss_bef = f['htmiss_before']
    htmiss_aft = f['htmiss_after']

    fig, ax = plt.subplots()

    hep.histplot(htmiss_bef.values, htmiss_bef.edges, ax=ax, label='Before rebalancing')
    hep.histplot(htmiss_aft.values, htmiss_aft.edges, ax=ax, label='After rebalancing')

    ax.set_xlabel(r'$H_T^{miss} \ (GeV)$', fontsize=14)
    ax.set_ylabel(r'Counts', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e7)

    ax.legend(title='Rebalancing')

    ax.text(0., 1., f'{tag_to_plottag(dataset_tag)} 2017',
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    # If we're looking at QCD and plot_gen=True, plot the GEN HTmiss distribution as well
    if dataset_tag == 'qcd' and plot_gen:
        # Coffea file to take GEN HT-miss distribution from
        accpath = './input/14Apr21/qcd_QCD_HT500to700-mg_2017.coffea'
        acc = load(accpath)

        distribution = 'gen_htmiss'
        h = acc[distribution].integrate('dataset').integrate('region', 'inclusive')

        hist.plot1d(h, ax=ax, clear=False)

    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == 'None':
            handle.set_label(r'GEN $H_T^{miss}$')

    ax.legend(handles=handles)

    outpath = pjoin(outdir, f'htmiss_before_after_reb.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    # Point the script to the directory containing the workspace files
    inpath = sys.argv[1]
    infiles = glob.glob(pjoin(inpath, 'ws_eventchunk*.root'))

    jobtag = re.findall('202\d.*', inpath)[0].split('/')[0]

    # Output directory for plotting
    outdir = f'./output/{jobtag}'

    # If the output directory exists, just take the existing ROOT file and do the plotting
    # Otherwise, first save the ROOT file with before/after distributions 
    if os.path.exists(outdir):
        outputrootpath = pjoin(outdir, 'htmiss_after_reb.root')
        print('INFO: ROOT file already exists, moving on to plotting')
    else:
        os.makedirs(outdir)
        outputrootpath = save_htmiss_before_and_after(infiles, outdir)

    dataset_tag = 'jetht' if re.match('.*[Jj]et[Hh][Tt].*', inpath) else 'qcd'

    # From the ROOT file created in previous step, plot the distributions
    # (uproot and matplotlib in the house)
    plot_htmiss_before_and_after(outdir, outputrootpath, dataset_tag)

if __name__ == '__main__':
    main()