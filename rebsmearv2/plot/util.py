#!/usr/bin/env python

import os
import sys
import re
import math
import yaml
import argparse
import numpy as np
import uproot
import uproot_methods.classes.TH1
import types

from collections import defaultdict
from matplotlib import pyplot as plt
from coffea import hist
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.dataset import is_data, extract_year
from pprint import pprint

pjoin = os.path.join

def load_xs():
    xsfile = rebsmear_path('data/xs.yml')
    with open(xsfile,'r') as f:
        xs_yml = yaml.load(f, Loader=yaml.FullLoader)

    xs = {dataset: _xs['gen'] for dataset, _xs in xs_yml.items()}
    
    return xs

def lumi(year):
    """Golden JSON luminosity per for given year
    :param year: Year of data taking
    :type year: int
    :return: Golden JSON luminosity for that year in pb (!)
    :rtype: float
    """
    if year==2018:
        return 59.7
    if year==2017:
        return 41.5
    if year==2016:
        return 35.9

def scale_xs_lumi_sumw(histogram, acc, scale_lumi=True):
    '''Scale the MC histograms with XS * lumi / sumw.'''
    # Get the list of datasets and filter MC data sets
    datasets = list(map(str, histogram.axis('dataset').identifiers()))

    mcs = [x for x in datasets if not is_data(x)]

    # Normalize to XS * lumi / sumw
    known_xs = load_xs()

    xs_map = {}
    sumw = defaultdict(float)
    for mc in mcs:
        try:
            ixs = known_xs[re.sub('_new_*pmx','',mc)]
        except KeyError:
            print(f"WARNING: Cross section not found for dataset {mc}. Using 0.")
            ixs = 0
        xs_map[mc] = ixs
        sumw[mc] = acc['sumw'][mc]

    norm_dict = {mc : 1e3 * xs_map[mc] * (lumi(extract_year(mc)) if scale_lumi else 1) / sumw[mc] for mc in mcs}
    histogram.scale(norm_dict, axis='dataset')

def rs_merge_datasets(histogram):
    '''Merge datasets that belong to the same physics process.'''
    all_datasets = list(map(str, histogram.identifiers('dataset')))
    mapping = {
        'JetHT_2017' : [x for x in all_datasets if re.match('JetHT_.*2017[A-Z]+',x)],
        'JetHT_2018' : [x for x in all_datasets if re.match('JetHT_.*2018[A-Z]+',x)],
        'QCD_HT_2017' : [x for x in all_datasets if re.match('QCD_HT.*_2017',x)],
        'QCD_HT_2018' : [x for x in all_datasets if re.match('QCD_HT.*_2018',x)],
    }

    # Apply the mapping
    histogram = histogram.group("dataset",hist.Cat("dataset", "Primary dataset"),  mapping)

    return histogram

class URTH1(uproot_methods.classes.TH1.Methods, list):
    def __init__(self, edges, sumw, sumw2, title=""):
        self._fXaxis = types.SimpleNamespace()
        self._fXaxis._fNbins = len(edges)-1
        self._fXaxis._fXmin = edges[0]
        self._fXaxis._fXmax = edges[-1]

        self._fXaxis._fXbins = edges.astype(">f8")

        centers = (edges[:-1] + edges[1:]) / 2.0
        self._fEntries = self._fTsumw = self._fTsumw2 = sumw[1:-1].sum()
        self._fTsumwx = (sumw * centers).sum()
        self._fTsumwx2 = (sumw * centers**2).sum()

        self._fName = title
        self._fTitle = title

        self.extend(sumw.astype(">f8"))
        self._classname = "TH1D"
        self._fSumw2 = sumw2.astype(">f8")