#!/usr/bin/env python

import os
import sys
import re
import math
import argparse
import numpy as np
import coffea.processor as processor
import uproot

from coffea.analysis_objects import JaggedCandidateArray
from coffea.processor.dataframe import LazyDataFrame
from coffea.util import save
from coffea import hist
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.dataset import is_data
from rebsmearv2.helpers.helpers import dphi
from pprint import pprint

pjoin = os.path.join

Bin = hist.Bin
Cat = hist.Cat
Hist = hist.Hist

def get_accumulator():
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")

    mjj_ax = Bin("mjj", r"$M_{jj}$ (GeV)", 150, 0, 7500)
    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50, -np.pi, np.pi)

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 100, 0, 4000)

    items = {}
    
    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)

    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

    items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi1"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

    items["mjj"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
    items["ht"] = Hist("Counts", dataset_ax, region_ax, ht_ax)
    items["htmiss"] = Hist("Counts", dataset_ax, region_ax, ht_ax)

    return processor.dict_accumulator(items)

class RSPostProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = get_accumulator()
        self._setup_regions()

    @property
    def accumulator(self):
        return self._accumulator

    def _setup_candidates(self, df):
        '''Set up candidates (mainly jets).'''
        ak4 = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'],
            eta=df['Jet_eta'],
            phi=df['Jet_phi'],
            mass=0. * df['Jet_pt']
        )

        htmiss = df['HTmiss']
        ht = df['HT']

        return ak4, htmiss, ht

    def _setup_regions(self):
        '''Set up selection regions.'''
        self.regions = {}
        self.regions['sr_vbf'] = [
            'mjj',
            'detajj',
            'dphijj',
            'hemisphere',
            'leadak4_pt_eta',
            'trailak4_pt_eta',
        ]

    def process(self,df):
        '''Fill and save histograms.'''
        dataset = df['dataset']
        # Set up physics objects
        ak4, htmiss, ht = self._setup_candidates(df)

        # Leading jet pair
        diak4 = ak4[:,:2].distincts()

        df['mjj'] = diak4.mass.max()
        df['dphijj'] = dphi(diak4.i0.phi.min(), diak4.i1.phi.max())
        df['detajj'] = np.abs(diak4.i0.eta - diak4.i1.eta).max()

        selection = processor.PackedSelection()
        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)
        
        leadak4_pt_eta = (diak4.i0.pt > 80) & (np.abs(diak4.i0.eta) < 4.7)
        trailak4_pt_eta = (diak4.i1.pt > 40) & (np.abs(diak4.i1.eta) < 4.7)
        hemisphere = (diak4.i0.eta * diak4.i1.eta < 0).any()

        selection.add('mjj', df['mjj'] > 200)
        selection.add('detajj', df['detajj'] > 1.)
        selection.add('dphijj', df['dphijj'] < 1.5)
        selection.add('hemisphere', hemisphere)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        selection.add('trailak4_pt_eta', trailak4_pt_eta.any())

        output = self.accumulator.identity()
        if not df['is_data']:
            output['sumw'][dataset] +=  df['sumw']
            output['sumw2'][dataset] +=  df['sumw2']
        
        for region, cuts in self.regions.items():
            mask = selection.all(*cuts)

            def ezfill(name, **kwargs):
                    """Helper function to make filling easier."""
                    output[name].fill(
                                    dataset=dataset,
                                    region=region,
                                    **kwargs
                                    )

            # Fill histograms
            ezfill('ak4_pt0',     jetpt=diak4.i0.pt[mask].flatten())
            ezfill('ak4_eta0',    jeteta=diak4.i0.eta[mask].flatten())
            ezfill('ak4_phi0',    jetphi=diak4.i0.phi[mask].flatten())

            ezfill('ak4_pt1',     jetpt=diak4.i1.pt[mask].flatten())
            ezfill('ak4_eta1',    jeteta=diak4.i1.eta[mask].flatten())
            ezfill('ak4_phi1',    jetphi=diak4.i1.phi[mask].flatten())

            ezfill('mjj',      mjj=df['mjj'][mask])
            ezfill('ht',       ht=ht[mask])
            ezfill('htmiss',   ht=htmiss[mask])

        return output

    def postprocess(self, accumulator):
        return accumulator

class PostProcExecutor():
    '''
    Wrapper object to take a set of ROOT files and convert to coffea histograms.

    INPUT: Set of ROOT files containing event information stored as TTrees.
    OUTPUT: Set of coffea files containing histograms.
    '''
    def __init__(self, files):
        self.files = files
    
    def _read_sumw_sumw2(self, file):
        runs = uproot.open(file)['Runs']
        return runs['sumw'].array()[0], runs['sumw2'].array()[0]

    def _analyze_file(self, file, treename='Events', flatten=True):
        '''
        Analyze a single file. Reads the "Events" tree and converts it into a LazyDataFrame.
        This df is then passed onto the RSPostProcessor.
        '''
        t = uproot.open(file)[treename]
        df = LazyDataFrame(t, flatten=flatten)

        # Dataset name from the filename
        df['dataset'] = re.sub('_rebalanced_tree_(\d+).root', '', os.path.basename(file))
        df['is_data'] = is_data(df['dataset'])
        
        ichunk = re.findall('tree_(\d+).root', os.path.basename(file))[0]

        if not df['is_data']:
            df['sumw'], df['sumw2'] = self._read_sumw_sumw2(file)
        
        # Process the dataframe!
        processor_instance = RSPostProcessor()
        out = processor_instance.process(df)

        # Save the output file
        outpath = pjoin(self.outdir, f'rebsmear_{df["dataset"]}_{ichunk}.coffea')
        save(out, outpath)

    def set_output_dir(self, outdir):
        self.outdir = outdir

    def analyze_files(self):
        for file in self.files:
            self._analyze_file(file)

if __name__ == '__main__':
    testfiles = [
        rebsmear_path('output_test/QCD_HT700to1000-mg_new_pmx_2017_rebalanced_tree_1.root'),
        rebsmear_path('output_test/QCD_HT700to1000-mg_new_pmx_2017_rebalanced_tree_11.root'),
    ]
    e = PostProcExecutor(testfiles)
    e.analyze_files()