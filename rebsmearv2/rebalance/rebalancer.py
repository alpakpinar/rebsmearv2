#!/usr/bin/env python

import os
import re
import math
import time
import argparse
import multiprocessing
import numpy as np
import uproot
import ROOT as r
r.gSystem.Load('libRooFit')

from datetime import date
from array import array
from rebsmearv2.rebalance.objects import Jet, RebalanceWSFactory, JERLookup
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.dataset import is_data

pjoin = os.path.join

class RebalanceExecutor():
    '''
    Object to execute the rebalancing step.

    INPUT: Takes the set of files to be processed.
    OUTPUT: Produces ROOT files with rebalanced event information saved.
    '''
    def __init__(self, files, dataset, treename, test=False, jersource='jer_mc', eventfrac=1e-3):
        self.files = files
        self.dataset = dataset
        self.treename = treename
        # Test mode: Only run on the first 10 events from the first 5 files
        self.test = test
        # Set up the JER source: "jer_data" or "jer_mc"
        self.jersource = jersource
        # Event fraction: Process X% of the events in the set of files, defaults to 0.1%
        self.eventfrac = eventfrac

    def _read_jets(self, event, tree, ptmin=30, absetamax=5.0):
        n = event

        pt, phi, eta = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['pt','phi','eta'])
        
        # Return jet collection with pt/eta cuts (if provided)
        return [Jet(pt=ipt, phi=iphi, eta=ieta) for ipt, iphi, ieta in zip(pt, phi, eta) if ( (ipt > ptmin) and (np.abs(ieta) < absetamax) ) ]

    def _read_sumw_sumw2(self, infile):
        '''Returns sumw and sumw2 for MC, to be used for scaling during post-processing.'''
        t = infile['Runs']
        return t['genEventSumw'].array()[0], t['genEventSumw2'].array()[0] 

    def set_output_dir(self, outdir):
        self.outdir = outdir
        try:
            os.makedirs(self.outdir)
        except FileExistsError:
            pass

    def process_file(self, filepath):
        '''Process a single file.'''
        infile = uproot.open(filepath)
        tree = infile[self.treename]
        numevents = len(tree)

        # Extract dataset name from the input path
        datasetname = filepath.split('/')[-4]
        treename = filepath.split('/')[-1].replace('.root','')

        # Set up output ROOT file
        outpath = pjoin(self.outdir, f"{datasetname}_rebalanced_{treename}.root")
        f = r.TFile(outpath,"RECREATE")
    
        if not is_data(self.dataset):
            sumw, sumw2 = self._read_sumw_sumw2(infile)

            # "Runs" tree to save sumw and sumw2
            t_runs = r.TTree('Runs', 'Runs')
            arr_sumw = array('f', [sumw])
            arr_sumw2 = array('f', [sumw2])
            t_runs.Branch('sumw', arr_sumw, 'sumw/F')
            t_runs.Branch('sumw2', arr_sumw2, 'sumw2/F')
            t_runs.Fill()
            t_runs.Write()

        # Set up the output tree to be saved
        nJetMax = 15
        outtree = r.TTree('Events','Events')
        
        njet = array('i', [0])
        jet_pt = array('f',  [0.] * nJetMax)
        jet_eta = array('f', [0.] * nJetMax)
        jet_phi = array('f', [0.] * nJetMax)
        
        htmiss = array('f', [0.])
        ht = array('f', [0.])
    
        # Set up branches for the output ROOT file
        outtree.Branch('nJet', njet, 'nJet/I')
        outtree.Branch('Jet_pt', jet_pt, 'Jet_pt[nJet]/F')
        outtree.Branch('Jet_eta', jet_eta, 'Jet_eta[nJet]/F')
        outtree.Branch('Jet_phi', jet_phi, 'Jet_phi[nJet]/F')
    
        outtree.Branch('HTmiss', htmiss, 'HTmiss/F')
        outtree.Branch('HT', ht, 'HT/F')

        # Loop over the events: Rebalance
        num_events_to_run = math.ceil(self.eventfrac * numevents) 
        print('STARTING REBALANCING')
        print(f'Total number of events: {numevents}')
        print(f'Number of events to run on: {num_events_to_run}')
        for event in range(num_events_to_run):
            # In test mode, only run on first 10 events
            if self.test and event == 10:
                break
            
            jets = self._read_jets(event, tree)
            rbwsfac = RebalanceWSFactory(jets)
            # JER source, initiate the object and specify the JER input
            jer_evaluator = JERLookup()

            jer_evaluator.from_th1(rebsmear_path("./input/jer.root"), self.jersource)
            
            rbwsfac.set_jer_evaluator(jer_evaluator)
            try:
                rbwsfac.build()
            except RuntimeError as e:
                # If error is due to HT < 100 GeV, simply discard the event and continue the loop
                if 'HT bin' in str(e):
                    print('WARNING: HT < 100 GeV, skipping event.')
                    continue
                # Shoot, something else has happened
                else:
                    raise RuntimeError(e)
            
            ws = rbwsfac.get_ws()

            f.cd()
            
            # Rebalancing: Do the minimization
            m = r.RooMinimizer(ws.function("nll"))
            m.migrad()

            # Fill the array for the output tree
            numjets = int(ws.var('njets').getValV())
            njet[0] = numjets
            for idx in range(numjets):
                jet_pt[idx] = ws.var('gen_pt_{}'.format(idx)).getValV()
                jet_eta[idx] = ws.var('reco_eta_{}'.format(idx)).getValV()
                jet_phi[idx] = ws.var('reco_phi_{}'.format(idx)).getValV()
    
            htmiss[0] = ws.function('gen_htmiss_pt').getValV()
            ht[0] = ws.function('gen_ht').getValV()
    
            outtree.Fill()

        # Once we're done with events, save 'em
        f.cd()
        outtree.Write()

        return outpath

    def process(self):
        '''Process the list of files.'''
        output_files = []
        for idx, filepath in enumerate(self.files):
            if self.test and idx == 5:
                break
            output_files.append(
                self.process_file(filepath)
            )
        # This returns a set of output files to be used in the next step.
        return output_files
