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
from pprint import pprint

pjoin = os.path.join

class RebalanceExecutor():
    '''
    Object to execute the rebalancing step.

    INPUT: Takes the set of files to be processed.
    OUTPUT: Produces ROOT files with rebalanced event information saved.
    '''
    def __init__(self, files, dataset, treename, test=False, jersource='jer_mc', eventfrac=1e-2):
        self.files = files
        self.dataset = dataset
        self.treename = treename
        # Test mode: Only run on the first 10 events from the first 5 files
        self.test = test
        # Set up the JER source: "jer_data" or "jer_mc"
        self.jersource = jersource
        # Event fraction: Process X% of the events in the set of files, defaults to 0.1%
        self.eventfrac = eventfrac

    def _jet_preselection(self, jet, ptmin=30, absetamax=5.0):
        '''Pre-selection for jets. Includes loose ID requirement (with minimum pt requirement) + HF shape cuts.'''
        abseta = np.abs(jet.eta)
        # Loose jet ID
        if not (jet.jetid&2 == 2):
            return False

        if not (jet.pt > ptmin and abseta < absetamax):
            return False

        # HF shape cuts
        if not (jet.hfcss < 3):
            return False

        if (abseta > 2.99) and (abseta < 4.0) and (jet.pt > 80):
            if not (jet.sieie - jet.sipip < 0.02):
                return False
            if ((jet.sieie < 0.02) and (jet.sipip < 0.02)):
                return False

        elif (abseta >= 4.0) and (jet.pt > 80):  
            if not ((jet.sieie < 0.1) and (jet.sipip > 0.02)):
                return False

        return True

    def _read_jets(self, event, tree, ptmin=30, absetamax=5.0):
        n = event

        pt, phi, eta = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['pt','phi','eta'])
        
        jetid = tree['Jet_jetId'].array(entrystart=n, entrystop=n+1)[0]

        sieie, sipip, hfcss = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['hfsigmaEtaEta', 'hfsigmaPhiPhi', 'hfcentralEtaStripSize'])

        jets = []
        for idx in range(len(pt)):
            j = Jet(
                pt=pt[idx],
                phi=phi[idx],
                eta=eta[idx],
                jetid=jetid[idx],
                sieie=sieie[idx],
                sipip=sipip[idx],
                hfcss=hfcss[idx],
            )
            if not self._jet_preselection(j):
                continue

            jets.append(j)

        return jets

    def _event_has_electron(self, event, tree):
        '''Returns whether an event contains a electron.'''
        n = event
        nelectron = tree['nElectron'].array(entrystart=n, entrystop=n+1)[0]
        if nelectron == 0:
            return False
        
        # Check for good quality electrons
        electron_pt = tree['Electron_pt'].array(entrystart=n, entrystop=n+1)[0]
        electron_eta = tree['Electron_eta'].array(entrystart=n, entrystop=n+1)[0]
        electron_detasc = tree['Electron_deltaEtaSC'].array(entrystart=n, entrystop=n+1)[0]
        electron_cutBased = tree['Electron_cutBased'].array(entrystart=n, entrystop=n+1)[0]

        def pass_cut(pt, eta, detasc, cutBased):
            return (np.abs(eta + detasc) < 2.5) and (pt > 10) and cutBased >= 1

        for info in zip(electron_pt, electron_eta, electron_detasc, electron_cutBased):
            if pass_cut(*info):
                return True             

        return False

    def _event_has_muon(self, event, tree):
        '''Returns whether an event contains a muon.'''
        n = event
        nmuon = tree['nMuon'].array(entrystart=n, entrystop=n+1)[0]
        if nmuon == 0:
            return False

        # Check for good quality muons
        muon_pt = tree['Muon_pt'].array(entrystart=n, entrystop=n+1)[0]
        muon_eta = tree['Muon_eta'].array(entrystart=n, entrystop=n+1)[0]
        muon_iso = tree['Muon_pfRelIso04_all'].array(entrystart=n, entrystop=n+1)[0]
        muon_looseId = tree['Muon_looseId'].array(entrystart=n, entrystop=n+1)[0]

        def pass_cut(pt, eta, iso, looseid):
            return (iso < 0.15) and (np.abs(eta) < 2.4) and (pt > 20) and looseid

        for info in zip(muon_pt, muon_eta, muon_iso, muon_looseId):
            if pass_cut(*info):
                return True             

        return False

    def _event_has_photon(self, event, tree):
        '''Returns whether an event contains a photon.'''
        n = event
        nphoton = tree['nPhoton'].array(entrystart=n, entrystop=n+1)[0]
        if nphoton == 0:
            return False

        # Check for good quality photons
        photon_pt = tree['Photon_pt'].array(entrystart=n, entrystop=n+1)[0]
        photon_eta = tree['Photon_eta'].array(entrystart=n, entrystop=n+1)[0]
        photon_cutBasedId = tree['Photon_cutBased'].array(entrystart=n, entrystop=n+1)[0]
        photon_electronVeto = tree['Photon_electronVeto'].array(entrystart=n, entrystop=n+1)[0]

        def pass_cut(pt, eta, cutBasedId, eleVeto):
            return (pt > 15) and (np.abs(eta) < 2.5) and cutBasedId >= 1 and eleVeto

        for info in zip(photon_pt, photon_eta, photon_cutBasedId, photon_electronVeto):
            if pass_cut(*info):
                return True

        return False

    def _event_has_tau(self, event, tree):
        '''Returns whether an event contains a tau.'''
        n = event
        ntau = tree['nTau'].array(entrystart=n, entrystop=n+1)[0]
        if ntau == 0:
            return False

        # Check for good quality taus
        tau_pt = tree['Tau_pt'].array(entrystart=n, entrystop=n+1)[0]
        tau_eta = tree['Tau_eta'].array(entrystart=n, entrystop=n+1)[0]
        tau_id = tree['Tau_idDecayModeNewDMs'].array(entrystart=n, entrystop=n+1)[0]
        tau_iso = tree['Tau_idDeepTau2017v2p1VSjet'].array(entrystart=n, entrystop=n+1)[0]
        
        def pass_cut(pt, eta, id, iso):
            return (pt > 20) and (np.abs(eta) < 2.3) and id and (iso & 2 == 2)

        for info in zip(tau_pt, tau_eta, tau_id, tau_iso):
            if pass_cut(*info):
                return True

        return False

    def _event_contains_lepton(self, event, tree):
        '''Returns whether an event contains a lepton or photon.'''
        if self._event_has_muon(event, tree):
            return True
        if self._event_has_photon(event, tree):
            return True
        if self._event_has_tau(event, tree):
            return True
        if self._event_has_electron(event, tree):
            return True

        return False

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
            
            # Check if the event contains a lepton or a photon, if so, veto the event
            if self._event_contains_lepton(event, tree):
                continue

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
            if self.test and idx == 1:
                break
            output_files.append(
                self.process_file(filepath)
            )
        # This returns a set of output files to be used in the next step.
        return output_files
