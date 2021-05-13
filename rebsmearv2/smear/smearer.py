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
from rebsmearv2.rebalance.objects import Jet, RebalanceWSFactory, JERLookup
from rebsmearv2.helpers.paths import rebsmear_path
from rebsmearv2.helpers.dataset import is_data

pjoin = os.path.join

class Smearer():
    '''
    Object for execution of the smearing module.
    
    INPUT: Takes the set of files to be processed (output of the rebalancing module). 
    OUTPUT: Produces ROOT files with rebalanced+smeared event information saved.
    '''
    def __init__(self):
        pass