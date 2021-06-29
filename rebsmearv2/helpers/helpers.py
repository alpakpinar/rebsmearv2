#!/usr/bin/env python
import numpy as np
import pandas as pd
from rebsmearv2.helpers.paths import rebsmear_path

def dphi(phi1, phi2):
    """Calculates delta phi between objects"""
    x = np.abs(phi1 - phi2)
    sign = x<=np.pi
    dphi = sign* x + ~sign * (2*np.pi - x)
    return dphi

def min_dphi_jet_met(jets, met_phi, ptmin=30, njet=4):
    jets = jets[(jets.pt > ptmin)]
    jets = jets[:,:njet]

    return dphi(jets.phi, met_phi).min()

def calc_mjj(jet1, jet2):
    '''Calculate the invariant mass of two jets.'''
    deta = np.cosh(jet1.eta - jet2.eta)
    dphi = np.cos(jet1.phi - jet2.phi)

    return np.sqrt(2 * jet1.pt * jet2.pt * (deta - dphi))

def dataframe_for_trigger_prescale(trigger):
    '''Initiates the dataframe for trigger prescale weights.'''
    filepath = f'./input/prescale/2017/prescales_{trigger}.csv'
    df = pd.read_csv(filepath)
    return df