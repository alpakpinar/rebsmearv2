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

from datetime import date, datetime
from array import array
from scipy.stats import expon, rv_histogram
from rebsmearv2.rebalance.objects import Jet, RebalanceWSFactory, JERLookup
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.dataset import is_data
from rebsmearv2.helpers.helpers import dphi, calc_mjj, dataframe_for_trigger_prescale
from pprint import pprint

pjoin = os.path.join

def decide_trigger_for_prescale(trigger_results):
    '''Given the trigger results for the event, determine the trigger with the highest threshold (lowest prescale).'''
    def decide_trigger_order(trigger):
        return int(re.findall('\d+', trigger)[0])


    # Sort the triggers according to the pt thresholds (in descending order)
    sorted_triggers = sorted(trigger_results.keys(), key=decide_trigger_order, reverse=True)

    # Determine the highest one passing
    trigger_to_look = None

    for t in sorted_triggers:
        if trigger_results[t] == 1:
            trigger_to_look = t
            break
    
    if trigger_to_look is None:
        raise RuntimeError('No trigger has passed for this event, this should not happen!')
    
    return trigger_to_look

def compute_prescale_weight(trigger_to_look, run, luminosityBlock):
    '''Given the event and list of trigger results, determine the trigger an event passes with the highest threshold.'''
    df_ps = dataframe_for_trigger_prescale(trigger_to_look)
    w = df_ps.loc[(df_ps['Run'] == run) & (df_ps['LumiSection'] == luminosityBlock)]['Prescale']
    return w.iloc[0], int(re.findall('\d+', trigger_to_look)[0])

def determine_second_lowest_prescale(trigger_results, trigger_thresh_for_ps_weight, run, luminosityBlock):
    '''
    If the event is passing multiple triggers, determine the second lowest prescale weight.
    If the event is only passing a single trigger, return -1.
    '''
    pslist = []
    for t,v in trigger_results.items():
        if v != 1:
            continue
        thresh = int(re.findall('\d+', t)[0])
        if thresh == trigger_thresh_for_ps_weight:
            continue
        pslist.append(
            compute_prescale_weight(t, run, luminosityBlock)[0]
        )
    
    if len(pslist) == 0:
        return -1
        
    return min(pslist)

class RebalanceExecutor():
    '''
    Object to execute the rebalancing step.

    INPUT: Takes the set of files to be processed.
    OUTPUT: Produces ROOT files with rebalanced event information saved.
    '''
    def __init__(self, 
            files,
            ichunk,
            dataset, 
            treename, 
            test=False, 
            jersource='jer_mc',
            htprescale=False,
            dphiprescale=False
            ):
        self.files = files
        self.ichunk = ichunk
        self.dataset = dataset
        self.treename = treename
        # Test mode: Only run on the first 10 events from the first 5 files
        self.test = test
        # Set up the JER source: "jer_data" or "jer_mc"
        self.jersource = jersource

        self.htprescale = htprescale
        self.dphiprescale = dphiprescale

    def _trigger_results(self, tree, event, triggers=['HLT_PFJet40']):
        '''Get a dictionary of values whether the given event passing each of the given triggers.'''
        mapping = {}
        n = event
        for trigname in triggers:
            trigval = tree[trigname].array(entrystart=n, entrystop=n+1)[0]
            mapping[trigname] = trigval

        return mapping

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

    def _kinematic_preselection(self, jets):
        '''Kinematic pre-selection on mjj and dphijj.'''
        mjj = calc_mjj(*jets[:2])
        if mjj < 100:
            return False

        htmiss = self._compute_htmiss(jets)
        # HTmiss > 150 GeV cut
        if htmiss < 150:
            return False

        return True

    def _read_jets(self, event, tree, ptmin=30, absetamax=5.0):
        n = event

        pt, phi, eta = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['pt','phi','eta'])
        
        jetid = tree['Jet_jetId'].array(entrystart=n, entrystop=n+1)[0]

        # HF shower shape variables
        sieie, sipip, hfcss = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['hfsigmaEtaEta', 'hfsigmaPhiPhi', 'hfcentralEtaStripSize'])

        # Jet energy fractions
        nef, nhf, cef, chf = (tree[f'Jet_{x}'].array(entrystart=n, entrystop=n+1)[0] for x in ['neEmEF', 'neHEF', 'chEmEF', 'chHEF'])

        jets = []
        for idx in range(len(pt)):
            # pt and eta requirement on jets
            jet_pass = (pt[idx] > ptmin) & (np.abs(eta[idx]) < absetamax)
            if not jet_pass:
                continue

            j = Jet(
                pt=pt[idx],
                phi=phi[idx],
                eta=eta[idx],
                jetid=jetid[idx],
                sieie=sieie[idx],
                sipip=sipip[idx],
                hfcss=hfcss[idx],
                nef=nef[idx],
                nhf=nhf[idx],
                cef=cef[idx],
                chf=chf[idx],
            )
            if not self._jet_preselection(j):
                continue

            jets.append(j)

        return jets

    def _compute_ht(self, jets):
        '''From the given jet collection, compute the HT of the event.'''
        return sum([j.pt for j in jets])

    def _compute_htmiss(self, jets):
        '''From the given jet collection, compute the HTmiss of the event.'''
        htmiss_x = 0
        htmiss_y = 0
        for j in jets:
            htmiss_x += j.px
            htmiss_y += j.py
        
        return np.hypot(htmiss_x, htmiss_y)

    def _compute_ht_prescale(self, ht):
        '''Based on the HT of the event before rebalancing, get the prescaling factor.'''
        if (ht > 100) & (ht < 300):
            return 100
        elif (ht > 300) & (ht < 500):
            return 10
        return 1

    def _compute_dphi_prescale(self, jets):
        '''Based on the dphi between the two leading jets of the event, get the prescaling factor.'''
        dphijj = dphi(jets[0].phi, jets[1].phi)

        if (dphijj > 2.5):
            return 100
        elif (dphijj > 2.) & (dphijj < 2.5):
            return 10
        return 1

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

        # Set up output ROOT file
        outpath = pjoin(self.outdir, f"{datasetname}_rebalanced_tree_{self.ichunk}.root")
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
        nJetMax = 20
        outtree = r.TTree('Events','Events')

        run = array('f', [0])
        luminosityBlock = array('f', [0])
        eventnum = array('f', [0])

        triggers = [
            'HLT_PFJet40',
            'HLT_PFJet60',
            'HLT_PFJet80',
            'HLT_PFJet140',
            'HLT_PFJet200',
            'HLT_PFJet260',
            'HLT_PFJet320',
            'HLT_PFJet400',
            'HLT_PFJet500',
            'HLT_PFJet550',
        ]

        njet = array('i', [0])
        jet_pt = array('f',  [0.] * nJetMax)
        jet_eta = array('f', [0.] * nJetMax)
        jet_phi = array('f', [0.] * nJetMax)
        
        jet_sieie = array('f',  [0.] * nJetMax)
        jet_sipip = array('f',  [0.] * nJetMax)
        jet_hfcss = array('f',  [0.] * nJetMax)

        jet_nef = array('f',  [0.] * nJetMax)
        jet_nhf = array('f',  [0.] * nJetMax)
        jet_cef = array('f',  [0.] * nJetMax)
        jet_chf = array('f',  [0.] * nJetMax)

        htmiss = array('f', [0.])
        ht = array('f', [0.])
        # Weight for HT based prescaling
        weight = array('f', [0.])
        # Weight for trigger prescaling
        weight_trigger_prescale = array('f', [0.])
        second_to_first_prescale_ratio = array('f', [0.])
    
        # Store the trigger that is used for prescale weight calculation
        trigger_thresh_for_ps = array('i', [0])

        # Set up branches for the output ROOT file
        outtree.Branch('run', run, 'run/I')
        outtree.Branch('luminosityBlock', luminosityBlock, 'luminosityBlock/I')
        outtree.Branch('event', eventnum, 'event/I')
        
        outtree.Branch('nJet', njet, 'nJet/I')
        outtree.Branch('Jet_pt', jet_pt, 'Jet_pt[nJet]/F')
        outtree.Branch('Jet_eta', jet_eta, 'Jet_eta[nJet]/F')
        outtree.Branch('Jet_phi', jet_phi, 'Jet_phi[nJet]/F')
    
        outtree.Branch('Jet_hfsigmaEtaEta', jet_sieie, 'Jet_hfsigmaEtaEta[nJet]/F')
        outtree.Branch('Jet_hfsigmaPhiPhi', jet_sipip, 'Jet_hfsigmaPhiPhi[nJet]/F')
        outtree.Branch('Jet_hfcentralEtaStripSize', jet_hfcss, 'Jet_hfcentralEtaStripSize[nJet]/F')
        
        outtree.Branch('Jet_neEmEF', jet_nef, 'Jet_neEmEF[nJet]/F')
        outtree.Branch('Jet_chEmEF', jet_cef, 'Jet_chEmEF[nJet]/F')
        outtree.Branch('Jet_neHEF', jet_nhf, 'Jet_neHEF[nJet]/F')
        outtree.Branch('Jet_chHEF', jet_chf, 'Jet_chHEF[nJet]/F')

        outtree.Branch('HTmiss', htmiss, 'HTmiss/F')
        outtree.Branch('HT', ht, 'HT/F')
        outtree.Branch('weight', weight, 'weight/F')
        outtree.Branch('weight_trigger_prescale', weight_trigger_prescale, 'weight_trigger_prescale/F')
        outtree.Branch('second_to_first_prescale_ratio', second_to_first_prescale_ratio, 'second_to_first_prescale_ratio/F')
        outtree.Branch('trigger_thresh_for_ps', trigger_thresh_for_ps, 'trigger_thresh_for_ps/I')

        # Initialize trigger branches (P/F)
        trig_arrays = {}
        for t in triggers:
            trig_arrays[t] = array('i', [0])
            outtree.Branch(t, trig_arrays[t], f'{t}/I')

        # Loop over the events: Rebalance
        print('STARTING REBALANCING')
        print(f'Total number of events: {numevents}')
        time_init = datetime.now()
        print(f'Time: {time_init}')
        for event in range(numevents):
            # In test mode, only run on first 1000 events
            if self.test and event == 100:
                break
            
            if event % 1e4 == 0:
                print(f'Processing event: {event}')
                print(f'Time: {datetime.now() - time_init}')

            # Trigger selection

            trigger_results = self._trigger_results(tree, event, triggers=triggers)
            # At least should pass one trigger
            if not any(list(trigger_results.values())):
                continue

            for t, v in trigger_results.items():
                trig_arrays[t][0] = v

            # Check if the event contains a lepton or a photon, if so, veto the event
            if self._event_contains_lepton(event, tree):
                continue


            jets = self._read_jets(event, tree)

            # Require a minimum of two jets (VBF phase space)
            if len(jets) < 2:
                continue
            
            # Kinematic pre-selection based on jets
            if not self._kinematic_preselection(jets):
                continue

            # Generate a random number between [0,1].
            # Use it to discard some events with low HT (private prescaling).
            randnum = np.random.rand()

            # Compute the HT and HTmiss of the event before rebalancing
            ht_bef = self._compute_ht(jets)
            # HT > 100 GeV cut
            if ht_bef < 100:
                continue

            # Prescale value based on HT
            prescale = None
            if self.htprescale:
                prescale = self._compute_ht_prescale(ht_bef)
                if randnum > (1/prescale):
                    continue

            # Prescale value based on dphi between the two leading jets
            if self.dphiprescale:
                prescale = self._compute_dphi_prescale(jets)
                if randnum > (1/prescale):
                    continue

            rbwsfac = RebalanceWSFactory(jets)
            # JER source, initiate the object and specify the JER input
            jer_evaluator = JERLookup()

            jer_evaluator.from_th1(rebsmear_path("./input/jer.root"), self.jersource)
            
            rbwsfac.set_jer_evaluator(jer_evaluator)
            rbwsfac.build()
            
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

                # Fill in more kinematic variables
                jet_sieie[idx] = jets[idx].sieie
                jet_sipip[idx] = jets[idx].sipip
                jet_hfcss[idx] = jets[idx].hfcss

                jet_nef[idx] = jets[idx].nef
                jet_nhf[idx] = jets[idx].nhf
                jet_cef[idx] = jets[idx].cef
                jet_chf[idx] = jets[idx].chf
    
            htmiss[0] = ws.function('gen_htmiss_pt').getValV()
            ht[0] = ws.function('gen_ht').getValV()
    
            # Compute the trigger prescale weight for this event
            run[0] = tree['run'].array(entrystart=event, entrystop=event+1)[0]
            luminosityBlock[0] = tree['luminosityBlock'].array(entrystart=event, entrystop=event+1)[0]
            eventnum[0] = tree['event'].array(entrystart=event, entrystop=event+1)[0]

            # Here's how we'll determine the prescale weights:
            # For each event, determine the trigger with the highest threshold for which it is passing
            # Read the PS values as a function of (run,lumi) for that trigger
            trigger_to_look = decide_trigger_for_prescale(trigger_results)
            ps_weight, trigger_thresh_for_ps_weight = compute_prescale_weight(trigger_to_look, run[0], luminosityBlock[0])
            
            weight_trigger_prescale[0] = ps_weight
            trigger_thresh_for_ps[0] = trigger_thresh_for_ps_weight

            # Look for the second smallest prescale weight for events passing multiple triggers.
            second_lowest_ps = determine_second_lowest_prescale(trigger_results, trigger_thresh_for_ps_weight, run[0], luminosityBlock[0])
            if second_lowest_ps == -1:
                second_to_first_prescale_ratio[0] = -1
            else:
                second_to_first_prescale_ratio[0] = second_lowest_ps / ps_weight
    
            # Store the prescale weight for later use
            # If no prescaling was done, weight would be just 1.
            if prescale:
                weight[0] = prescale
            else:
                weight[0] = 1

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
