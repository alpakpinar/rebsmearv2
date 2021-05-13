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
from pprint import pprint

pjoin = os.path.join

Bin = hist.Bin
Cat = hist.Cat
Hist = hist.Hist

def get_accumulator():
    dataset_ax = Cat("dataset", "Primary dataset")

    mjj_ax = Bin("mjj", r"$M_{jj}$ (GeV)", 150, 0, 7500)
    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50, -np.pi, np.pi)

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 100, 0, 4000)

    items = {}
    
    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)

    items["ak4_pt0"] = Hist("Counts", dataset_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, jet_phi_ax)

    items["ak4_pt1"] = Hist("Counts", dataset_ax, jet_pt_ax)
    items["ak4_eta1"] = Hist("Counts", dataset_ax, jet_eta_ax)
    items["ak4_phi1"] = Hist("Counts", dataset_ax, jet_phi_ax)

    items["mjj"] = Hist("Counts", dataset_ax, mjj_ax)
    items["ht"] = Hist("Counts", dataset_ax, ht_ax)
    items["htmiss"] = Hist("Counts", dataset_ax, ht_ax)

    return processor.dict_accumulator(items)

class RSPostProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = get_accumulator()

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

    def process(self,df):
        '''Fill and save histograms.'''
        dataset = df['dataset']
        # Set up physics objects
        ak4, htmiss, ht = self._setup_candidates(df)

        # Leading jet pair
        diak4 = ak4[:,:2].distincts()

        df['mjj'] = diak4.mass.max()

        output = self.accumulator.identity()
        if not df['is_data']:
            output['sumw'][dataset] +=  df['sumw']
            output['sumw2'][dataset] +=  df['sumw2']
        
        def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  **kwargs
                                  )

        # Fill histograms
        ezfill('ak4_pt0',     jetpt=diak4.i0.pt.flatten())
        ezfill('ak4_eta0',    jeteta=diak4.i0.eta.flatten())
        ezfill('ak4_phi0',    jetphi=diak4.i0.phi.flatten())

        ezfill('ak4_pt1',     jetpt=diak4.i1.pt.flatten())
        ezfill('ak4_eta1',    jeteta=diak4.i1.eta.flatten())
        ezfill('ak4_phi1',    jetphi=diak4.i1.phi.flatten())

        ezfill('mjj',    mjj=df['mjj'])
        ezfill('ht',     ht=ht)
        ezfill('htmiss',     ht=htmiss)

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