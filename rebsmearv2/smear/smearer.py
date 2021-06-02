#!/usr/bin/env python

import os
import re
import math
import time
import argparse
import numpy as np
import uproot
import coffea.processor as processor
import ROOT as r

from awkward import JaggedArray
from coffea.analysis_objects import JaggedCandidateArray
from coffea.processor.dataframe import LazyDataFrame
from coffea.lookup_tools import extractor
from coffea.util import save
from coffea import hist

from datetime import date
from array import array
from rebsmearv2.rebalance.objects import Jet, JERLookup
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.helpers import dphi, min_dphi_jet_met
from rebsmearv2.helpers.dataset import is_data

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

def create_evaluator_forJER(jersource):
    '''
    Create lookup object for jet energy resolution.
    JER source: jer_data or jer_mc
    '''
    ext = extractor()

    inputpath = rebsmear_path('input/jer.root')
    ext.add_weight_sets([f'jer_sigma {jersource} {inputpath}'])

    ext.finalize()
    evaluator = ext.make_evaluator()
    return evaluator

def read_resolution(ak4, jersource):
    '''
    Read the resolution for each jet in each event.
    Returns a Jagged array of resolution values (sigma) per jet.
    JER source: jer_data or jer_mc
    '''
    evaluator = create_evaluator_forJER(jersource)
    sigma = evaluator['jer_sigma'](ak4.pt, ak4.eta)
    return sigma

class CoffeaSmearer(processor.ProcessorABC):
    def __init__(self, ntoys=100, jersource='jer_mc'):
        self._accumulator = get_accumulator()
        self._setup_regions()
        self.ntoys = ntoys
        self.jersource = jersource

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

        return ak4

    def _setup_regions(self):
        '''Set up selection regions.'''
        self.regions = {}
        self.regions['inclusive'] = ['inclusive']
        common_cuts = [
            'inclusive',
            'mjj',
            'detajj',
            'dphijj',
            'hemisphere',
            'leadak4_pt_eta',
            'trailak4_pt_eta',
            'htmiss'
        ]
        # Two regions: VBF-like signal region and a QCD CR
        self.regions['sr_vbf'] = common_cuts + ['dphijm_sr']
        self.regions['cr_vbf_qcd'] = common_cuts + ['dphijm_cr']

    def do_smear(self, ak4):
        '''
        Core function to do the smearing and return the modified jet objects.
        
        For each jet, the resolution value will be read from the JER input, and for each event,
        the smearing will be applied N_toys times, each resulting in a different event definition.
        
        In the returned object, there will be N_toys x N_events # of events stored.
        '''
        sigma = read_resolution(ak4, self.jersource).flatten()

        # Generate Ntoys number of random numbers for each jet
        rand = np.array(
            [np.random.normal(1., s, self.ntoys) for s in sigma]
        ) 
        # randarr[:,:,k] gives the smear factors for each jet for toy # k 
        randarr = JaggedArray.fromcounts(ak4.counts, rand)

        # smeared_pt contains a set of smeared jet pts for Ntoys x Nevents # of events.
        # smeared_pt[k] will retrieve the pt values for event # k
        smeared_pt = np.concatenate(
            [randarr[:, :, k] * ak4.pt for k in range(self.ntoys)]
        )

        # For eta and phi values, simply iterate them for each toy since they won't be changing with
        jeteta = np.concatenate(
            [ak4.eta for k in range(self.ntoys)]
        )
        jetphi = np.concatenate(
            [ak4.phi for k in range(self.ntoys)]
        )

        jagged_jet_pt = JaggedArray.fromiter(smeared_pt)
        njets = jagged_jet_pt.counts
        
        jetpt = jagged_jet_pt.flatten()
        jeteta = JaggedArray.fromiter(jeteta).flatten()
        jetphi = JaggedArray.fromiter(jetphi).flatten()
        
        # Here we go, construct the new jet objects with these new jagged arrays!
        newak4 = JaggedCandidateArray.candidatesfromcounts(
            njets,
            pt=jetpt,
            eta=jeteta,
            phi=jetphi,
            mass=jetpt * 0.
        )

        return newak4

    def process(self, df):
        dataset = df['dataset']
        # Set up physics objects
        ak4 = self._setup_candidates(df)
        if ak4.size == 0:
            return
        # Get the smeared jets: This will contain a set of Nevents x Ntoys # of events
        sak4 = self.do_smear(ak4)

        # Leading jet pair
        diak4 = sak4[:,:2].distincts()

        mjj = diak4.mass.max()
        dphijj = dphi(diak4.i0.phi.min(), diak4.i1.phi.max())
        detajj = np.abs(diak4.i0.eta - diak4.i1.eta).max()

        selection = processor.PackedSelection()
        pass_all = np.ones(len(mjj))==1
        selection.add('inclusive', pass_all)
        
        # Implement VBF cuts
        leadak4_pt_eta = (diak4.i0.pt > 80) & (np.abs(diak4.i0.eta) < 4.7)
        trailak4_pt_eta = (diak4.i1.pt > 40) & (np.abs(diak4.i1.eta) < 4.7)
        hemisphere = (diak4.i0.eta * diak4.i1.eta < 0).any()

        selection.add('mjj', mjj > 200)
        selection.add('detajj', detajj > 1.)
        selection.add('dphijj', dphijj < 1.5)
        selection.add('hemisphere', hemisphere)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        selection.add('trailak4_pt_eta', trailak4_pt_eta.any())

        # With the smeared jets, recompute HT & HTmiss (equivalently, MET)
        ht = sak4[sak4.pt>30].pt.sum()
        htmiss = sak4[sak4.pt>30].p4.sum().pt

        met_phi = sak4[sak4.pt>30].p4.sum().phi

        # Require large HTmiss
        selection.add('htmiss', htmiss > 250)

        # Dphi(jet,MET) > 0.5 for SR
        # Dphi(jet,MET) < 0.5 for QCD CR
        dphijm = min_dphi_jet_met(sak4, met_phi)
        selection.add('dphijm_sr', dphijm > 0.5)
        selection.add('dphijm_cr', dphijm < 0.5)

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

            ezfill('mjj',      mjj=mjj[mask])
            ezfill('ht',       ht=ht[mask])
            ezfill('htmiss',   ht=htmiss[mask])

        return output

    def postprocess(self, accumulator):
        return accumulator

class SmearExecutor():
    '''
    Wrapper object to take a set of ROOT files, apply smearing to events and
    save data to coffea histograms.

    INPUT: Set of ROOT files containing event information stored as TTrees.
    OUTPUT: Set of coffea files containing histograms with smeared events.
    '''
    def __init__(self, files, ntoys=50):
        self.files = files
        # Number of toys, this many events will be generated per rebalanced event
        # (1 event = 1 smearing)
        self.ntoys = ntoys
    
    def _read_sumw_sumw2(self, file):
        runs = uproot.open(file)['Runs']
        return runs['sumw'].array()[0], runs['sumw2'].array()[0]    

    def set_output_dir(self, outdir):
        self.outdir = outdir

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

        processor_instance = CoffeaSmearer(ntoys=self.ntoys)
        out = processor_instance.process(df)

        # Save the output file
        outpath = pjoin(self.outdir, f'rebsmear_{df["dataset"]}_{ichunk}.coffea')
        save(out, outpath)

    def analyze_files(self):
        for file in self.files:
            self._analyze_file(file)