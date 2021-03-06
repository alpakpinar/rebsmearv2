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
from rebsmearv2.helpers.helpers import dphi, min_dphi_jet_met, dataframe_for_trigger_prescale
from rebsmearv2.helpers.dataset import is_data
from tqdm import tqdm

pjoin = os.path.join

Bin = hist.Bin
Cat = hist.Cat
Hist = hist.Hist

def accu_int():
    return processor.defaultdict_accumulator(int)

def empty_column_accumulator_float16():
    return processor.column_accumulator(np.array([],dtype=np.float16))

def get_accumulator(regions):
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")
    trigger_ax = Cat("trigger", "Trigger for prescale")

    mjj_ax = Bin("mjj", r"$M_{jj}$ (GeV)", 150, 0, 7500)
    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50, -np.pi, np.pi)
    frac_ax = Bin("frac", r"Jet Energy Fraction", 50, 0, 1)

    sieie_ax = Bin("sieie", r"$\sigma_{\eta\eta}$", 50, 0, 0.5)
    sipip_ax = Bin("sipip", r"$\sigma_{\phi\phi}$", 50, 0, 0.5)

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 100, 0, 4000)
    mht_ax = Bin("ht", r"$H_{T}$ (GeV)", 80, 0, 800)
    dphi_ax = Bin("dphi", r"$\Delta\phi$", 50, 0, 3.5)


    items = {}
    
    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)

    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_nef0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_cef0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_chf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)

    items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi1"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_nef1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_cef1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_chf1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)

    items["dphijm"] = Hist("Counts", dataset_ax, region_ax, dphi_ax)
    items["mjj"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
    items["ht"] = Hist("Counts", dataset_ax, region_ax, ht_ax)
    items["htmiss"] = Hist("Counts", dataset_ax, region_ax, mht_ax)

    items['mjj_per_trigger'] = Hist("Counts", dataset_ax, region_ax, mjj_ax, trigger_ax)
    items['ak4_phi0_per_trigger'] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax, trigger_ax)
    items['ak4_phi1_per_trigger'] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax, trigger_ax)

    items['high_ps_events'] = processor.defaultdict_accumulator(empty_column_accumulator_float16)


    for region in regions:
        if region == "inclusive":
            continue
        items[f'cutflow_{region}']  = processor.defaultdict_accumulator(accu_int)

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
    def __init__(self, eventweight, ntoys, jersource='jer_mc'):
        self._setup_regions()
        self._accumulator = get_accumulator(self.regions)
        self.ntoys = ntoys
        self.jersource = jersource
        self.eventweight = eventweight

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
            mass=0. * df['Jet_pt'],
            nef=df['Jet_neEmEF'],
            cef=df['Jet_chEmEF'],
            nhf=df['Jet_neHEF'],
            chf=df['Jet_chHEF'],
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

    def do_smear(self, ak4, weight):
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

        # Propagate the energy fractions
        jet_nef = np.concatenate(
            [ak4.nef for k in range(self.ntoys)]
        )
        jet_nhf = np.concatenate(
            [ak4.nhf for k in range(self.ntoys)]
        )
        jet_cef = np.concatenate(
            [ak4.cef for k in range(self.ntoys)]
        )
        jet_chf = np.concatenate(
            [ak4.chf for k in range(self.ntoys)]
        )

        jagged_jet_pt = JaggedArray.fromiter(smeared_pt)
        njets = jagged_jet_pt.counts
        
        jetpt = jagged_jet_pt.flatten()
        jeteta = JaggedArray.fromiter(jeteta).flatten()
        jetphi = JaggedArray.fromiter(jetphi).flatten()
        
        jetnef = JaggedArray.fromiter(jet_nef).flatten()
        jetnhf = JaggedArray.fromiter(jet_nhf).flatten()
        jetcef = JaggedArray.fromiter(jet_cef).flatten()
        jetchf = JaggedArray.fromiter(jet_chf).flatten()
        
        # Here we go, construct the new jet objects with these new jagged arrays!
        newak4 = JaggedCandidateArray.candidatesfromcounts(
            njets,
            pt=jetpt,
            eta=jeteta,
            phi=jetphi,
            mass=jetpt * 0,
            nef=jetnef,
            nhf=jetnhf,
            cef=jetcef,
            chf=jetchf,
        )

        # Also return a modified event weight array
        weights = np.concatenate(
            [weight for k in range(self.ntoys)]
        )

        return newak4, weights

    def process(self, df):
        dataset = df['dataset']
        # Set up physics objects
        ak4 = self._setup_candidates(df)
        if ak4.size == 0:
            return
        # Get the smeared jets: This will contain a set of Nevents x Ntoys # of events
        sak4, weight = self.do_smear(ak4, weight=self.eventweight)

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

        selection.add('mjj', mjj > 200.)
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
        
        # Calculte trigger prescale weight
        trigger_ps_weight = weight * self.ntoys

        for region, cuts in self.regions.items():

            # Fill cutflow
            if region != 'inclusive':
                output[f'cutflow_{region}'][dataset]['all'] += len(sak4)
                for icut, cutname in enumerate(cuts):
                    output['cutflow_' + region][dataset][cutname] += selection.all(*cuts[:icut+1]).sum()
            
            mask = selection.all(*cuts)

            # Save per-event values when the weight is large (in signal region)
            if region == 'sr_vbf':
                psmask = trigger_ps_weight[mask] > 1e5
                
                output['high_ps_events']['ht'] += processor.column_accumulator(np.float16(ht[mask][psmask]))
                output['high_ps_events']['htmiss'] += processor.column_accumulator(np.float16(htmiss[mask][psmask]))
                output['high_ps_events']['met_phi'] += processor.column_accumulator(np.float16(met_phi[mask][psmask]))
                output['high_ps_events']['mjj'] += processor.column_accumulator(np.float16(mjj[mask][psmask]))
                output['high_ps_events']['dphijm'] += processor.column_accumulator(np.float16(dphijm[mask][psmask]))
                
                output['high_ps_events']['ak4_pt0'] = processor.column_accumulator(np.float16(diak4.i0.pt[mask][psmask]))
                output['high_ps_events']['ak4_eta0'] = processor.column_accumulator(np.float16(diak4.i0.eta[mask][psmask]))
                output['high_ps_events']['ak4_pt1'] = processor.column_accumulator(np.float16(diak4.i1.pt[mask][psmask]))
                output['high_ps_events']['ak4_eta1'] = processor.column_accumulator(np.float16(diak4.i1.eta[mask][psmask]))
                
                output['high_ps_events']['ps_weight'] = processor.column_accumulator(np.float16(trigger_ps_weight[mask][psmask]))

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                dataset=dataset,
                                region=region,
                                **kwargs
                                )

            # Fill histograms
            ezfill('ak4_pt0',     jetpt=diak4.i0.pt[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_eta0',    jeteta=diak4.i0.eta[mask].flatten(),    weight=weight[mask])
            ezfill('ak4_phi0',    jetphi=diak4.i0.phi[mask].flatten(),    weight=weight[mask])
            ezfill('ak4_nef0',    frac=diak4.i0.nef[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_nhf0',    frac=diak4.i0.nhf[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_cef0',    frac=diak4.i0.cef[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_chf0',    frac=diak4.i0.chf[mask].flatten(),      weight=weight[mask])

            ezfill('ak4_pt1',     jetpt=diak4.i1.pt[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_eta1',    jeteta=diak4.i1.eta[mask].flatten(),    weight=weight[mask])
            ezfill('ak4_phi1',    jetphi=diak4.i1.phi[mask].flatten(),    weight=weight[mask])
            ezfill('ak4_nef1',    frac=diak4.i1.nef[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_nhf1',    frac=diak4.i1.nhf[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_cef1',    frac=diak4.i1.cef[mask].flatten(),      weight=weight[mask])
            ezfill('ak4_chf1',    frac=diak4.i1.chf[mask].flatten(),      weight=weight[mask])

            ezfill('mjj',      mjj=mjj[mask],        weight=weight[mask] )
            ezfill('ht',       ht=ht[mask],          weight=weight[mask] )
            ezfill('htmiss',   ht=htmiss[mask],      weight=weight[mask] )
            ezfill('dphijm',   dphi=dphijm[mask],    weight=weight[mask] )

            if region != 'inclusive':
                # Save histograms per trigger threshold
                thresholds = np.concatenate(
                    [df['trigger_thresh_for_ps'] for i in range(self.ntoys)]
                )
                trignamefunc = np.vectorize(lambda thresh: f'HLT_PFJet{thresh}')

                thresholds = JaggedArray.fromiter(thresholds).flatten()
                triggers = trignamefunc(thresholds)
                if len(triggers[mask]) > 0:
                    ezfill('mjj_per_trigger',        mjj=mjj[mask],   trigger=triggers[mask],   weight=weight[mask] )
                    ezfill('ak4_phi0_per_trigger',   jetphi=diak4.i0.phi[mask].flatten(),   trigger=triggers[mask],   weight=weight[mask] )
                    ezfill('ak4_phi1_per_trigger',   jetphi=diak4.i1.phi[mask].flatten(),   trigger=triggers[mask],   weight=weight[mask] )

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
    def __init__(self, files, ichunk, ntoys=int(1e3)):
        self.files = files
        self.ichunk = ichunk
        # Number of toys, this many events will be generated per rebalanced event
        # (1 event = 1 smearing)
        self.ntoys = ntoys
        # Event weight is calculated as (prescale weight HLT_PFJet40) / (num toys)
        # NOTE: Private prescaling weight will be added to this variable as we process
        self.weight_toys = 1 / ntoys

    def _read_sumw_sumw2(self, file):
        runs = uproot.open(file)['Runs']
        return runs['sumw'].array()[0], runs['sumw2'].array()[0]    

    def set_output_dir(self, outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.outdir = outdir

    def _analyze_file(self, file, treename='Events', chunksize=1000, flatten=True):
        '''
        Analyze a single file. Reads the "Events" tree and converts it into a LazyDataFrame.
        This df is then passed onto the RSPostProcessor.
        '''
        t = uproot.open(file)[treename]
        numevents = len(t)

        chunks = []
        if numevents % chunksize == 0:
            nchunks = numevents // chunksize
        else:
            nchunks = numevents // chunksize + 1
        
        for ichunk in range(nchunks):
            dfchunk = LazyDataFrame(t, 
                    entrystart=ichunk * chunksize, 
                    entrystop=(ichunk+1) * chunksize,
                    flatten=flatten
                    )
            chunks.append(dfchunk)
        
        for numchunk, df in enumerate(chunks):
            # Dataset name from the filename
            df['dataset'] = re.sub('_rebalanced_tree_(\d+).root', '', os.path.basename(file))
            df['is_data'] = is_data(df['dataset'])
            
            if not df['is_data']:
                df['sumw'], df['sumw2'] = self._read_sumw_sumw2(file)
    
            # Event weight is: PS weight / Ntoys
            eventweight = self.weight_toys * df['weight_trigger_prescale']
    
            processor_instance = CoffeaSmearer(eventweight=eventweight, ntoys=self.ntoys)
    
            out = processor_instance.process(df)
            # Save the output file
            outpath = pjoin(self.outdir, f'rebsmear_{df["dataset"]}_{self.ichunk}_{numchunk:03d}of{nchunks:03d}.coffea')
            save(out, outpath)

    def analyze_files(self):
        for file in tqdm(self.files):
            self._analyze_file(file)