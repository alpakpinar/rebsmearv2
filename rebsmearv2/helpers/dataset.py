#!/usr/bin/env python

import os
import re
import subprocess
from collections import defaultdict
from pprint import pprint

pjoin = os.path.join

def is_data(datasetname):
    return 'JetHT' in datasetname

def extract_year(dataset):
    for x in [6,7,8]:
        if f"201{x}" in dataset:
            return 2010+x
    raise RuntimeError("Could not determine dataset year")

def eosfind(path):
    cmd = ['eos', 'root://cmseos.fnal.gov/', 'find',  '--size', path]
    return subprocess.check_output(cmd).decode('utf-8')

def find_files_eos(directory, regex):
    fileset = defaultdict(list)
    lines = eosfind(re.sub('.*/store/','/store/',directory)).splitlines()
    for line in lines:
        parts = line.split()

        # Ignore lines representing directories
        if len(parts) < 2:
            continue
        # Ensure we are not missing a part
        if len(parts) > 2:
            raise RuntimeError(f'Encountered malformed line: {line}')

        # The info we care about
        path = parts[0].replace('path=','')
        if not path.endswith('.root'):
            continue

        dataset = path.split('/')[9]
        if not re.match(regex, dataset):
            continue
        fileset[dataset].append(re.sub('.*/store','root://cmsxrootd-site.fnal.gov//store', path))
    return fileset

def files_from_eos(regex):
    '''Generate file list per dataset from EOS.

    :param regex: Regular expression to match datasets
    :type regex: string
    :return: Mapping of dataset : [files]
    :rtype: dict
    '''
    topdir = '/eos/uscms/store/user/aakpinar/nanopost'
    tag = 'ULv8_05Feb21_rebsmear'
    fileset = find_files_eos(pjoin(topdir, tag), regex)
    
    return fileset