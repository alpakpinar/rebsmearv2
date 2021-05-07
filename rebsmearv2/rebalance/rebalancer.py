#!/usr/bin/env python

import os
import re
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

pjoin = os.path.join

class RebalanceExecutor():
    '''
    Object to execute the rebalancing step.

    INPUT: Takes the set of files to be processed.
    OUTPUT: Produces ROOT files with rebalanced event information saved.
    '''
    def __init__(self, files, dataset, treename, test=False, jersource='jer_mc'):
        self.files = files
        self.dataset = dataset
        self.treename = treename
        # Test mode: Only run on the first 10 events from the first 5 files
        self.test = test
        # Set up the JER source: "jer_data" or "jer_mc"
        self.jersource = jersource

    def _read_jets(self, event, tree, ptmin=30, absetamax=5.0):
        n = event

        pt, phi, eta = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['pt','phi','eta'])
        
        # Return jet collection with pt/eta cuts (if provided)
        return [Jet(pt=ipt, phi=iphi, eta=ieta) for ipt, iphi, ieta in zip(pt, phi, eta) if ( (ipt > ptmin) and (np.abs(ieta) < absetamax) ) ]

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
        outputrootdir = pjoin(self.outdir, datasetname)
        try:
            os.makedirs(outputrootdir)
        except FileExistsError:
            pass
        
        f = r.TFile(pjoin(outputrootdir, f"rebalanced_{treename}.root"),"RECREATE")
    
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
        for event in range(numevents):
            # In test mode, only run on first 10 events
            if self.test and event == 10:
                break
            
            jets = self._read_jets(event, tree)
            rbwsfac = RebalanceWSFactory(jets)
            # JER source, initiate the object and specify the JER input
            jer_evaluator = JERLookup()

            jer_evaluator.from_th1("./input/jer.root", self.jersource)
            
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

    def process(self):
        '''Process the list of files.'''
        for idx, filepath in enumerate(self.files):
            if self.test and idx == 5:
                break
            self.process_file(filepath)
