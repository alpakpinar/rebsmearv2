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

from datetime import date
from array import array
from rebsmearv2.rebalance.objects import Jet, JERLookup
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.dataset import is_data

pjoin = os.path.join

class Smearer():
    '''
    Object to apply smearing.
    Takes in one event, applies smearing to all the jets depending on (pt,eta), returns the event.
    '''
    def __init__(self, jer_evaluator):
        # JER evaluator object, already configured for the right histogram ("jer_data" or "jer_mc")
        # Should be passed in to the object, will be used to determine the SF 
        self._jer_evaluator = jer_evaluator
        
    def do_smear_on_jet(self, jet):
        '''
        Apply smearing on a single jet.
        RETURNS: Jet instance with smeared pt.
        '''
        # Determine the JER SF based on pt and eta
        sigma = self._jer_evaluator.get_jer(jet.pt, jet.eta)
        # TODO: Work out the jet pt update!
        # Update the object's transverse momentum
        jet.set_pt(sigma * jet.pt)
        return jet

    def do_smear(self, jets):
        '''Apply smearing on the event.'''
        for jet in jets:
            # Retrieve the smeared jet object
            mjet = self.do_smear_on_jet(jet)

        return jets

    def calculate_ht_htmiss(self, smeared_jets):
        '''With the smeared jets, re-calculate the HT and HTmiss quantities.'''
        njet = len(smeared_jets)
        ht = 0
        total_px = 0
        total_py = 0
        for jet in smeared_jets:
            ht += jet.pt
            total_px += jet.px
            total_py += jet.py

        htmiss = np.hypot(total_px, total_py)
        return ht, htmiss

class SmearingExecutor():
    '''
    Object for execution of the smearing module.
    
    INPUT: Takes the set of files to be processed (output of the rebalancing module). 
    OUTPUT: Produces ROOT files with rebalanced+smeared event information saved.
    '''
    def __init__(self, files, dataset, treename, jersource='jer_mc'):
        self.files = files
        self.dataset = dataset
        self.treename = treename
        # Set up the JER source: "jer_data" or "jer_mc"
        self.jersource = jersource
       
    def _read_sumw_sumw2(self, infile):
        '''Returns sumw and sumw2 for MC, to be used for scaling during post-processing.'''
        t = infile['Runs']
        return t['sumw'].array()[0], t['sumw2'].array()[0] 

    def _read_jets(self, event, tree, ptmin=30, absetamax=5.0):
        '''
        Returns a collection of Jet objects for a given event.
        '''
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
        base = os.path.basename(filepath)
        datasetname = re.sub('_rebalanced_tree_(\d+).root', '', base)
        treeindex = re.findall('tree_(\d+).root', base)[0]

        # Set up output ROOT file
        outpath = pjoin(self.outdir, f"{datasetname}_rebalanced_smeared_tree_{treeindex}.root")
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

        # Loop over the events: Smear
        for event in range(numevents):
            # Retrieve the rebalanced jets
            jets = self._read_jets(event, tree=tree)

            # JER source, initiate the object and specify the JER input
            jer_evaluator = JERLookup()

            jer_evaluator.from_th1(rebsmear_path("./input/jer.root"), self.jersource)

            # For each event, construct the Smearer object and do the smearing on each jet.
            # The object will return the modified jet pts so store them in the output tree.
            smearer = Smearer(jer_evaluator=jer_evaluator)

            # Retrieve the set of smeared jets
            smeared_jets = smearer.do_smear(jets)

            f.cd()
            # Update the arrays with jet pt/eta/phi information
            for j in smeared_jets:
                jet_pt.append(j.pt)
                jet_eta.append(j.eta)
                jet_phi.append(j.phi)
            
            numjets = len(smeared_jets)
            njet[0] = numjets

            ht[0], htmiss[0] = smearer.calculate_ht_htmiss(smeared_jets)

            outtree.Fill()

        # Once we're done with events, save 'em
        f.cd()
        outtree.Write()

        return outpath
    
    def process(self):
        '''Process the list of files.'''
        output_files = []
        for idx, filepath in enumerate(self.files):
            base = os.path.basename(filepath)
            output_files.append(
                self.process_file(filepath)
            )
        # This returns a set of output files to be used in the next step.
        return output_files
