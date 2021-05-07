#!/usr/bin/env python

import os
import sys
import re
import uproot
import ROOT as r
import numpy as np
from glob import glob
from tqdm import tqdm

pjoin = os.path.join


def save_delta_htmiss(infiles, outtag, outf, h_delta_htmiss):
    for inpath in tqdm(infiles):
        f = r.TFile(inpath, 'READ')

        events_before = [key.GetName() for key in f.GetListOfKeys() if key.GetName().startswith('before')] 
        events_after = [key.GetName() for key in f.GetListOfKeys() if key.GetName().startswith('rebalanced')]

        assert len(events_before) == len(events_after)
        nevents = len(events_before)

        for idx in range(nevents):
            ws_bef = f.Get(events_before[idx])
            ws_reb = f.Get(events_after[idx])

            htmiss_bef = ws_bef.function('gen_htmiss_pt').getValV()
            htmiss_reb = ws_reb.function('gen_htmiss_pt').getValV()

            delta_htmiss = np.abs(htmiss_bef - htmiss_reb) / htmiss_bef

            h_delta_htmiss.Fill(delta_htmiss)
    
    outf.cd()
    outf.Write()

def main():
    # Input directory containing workspace files
    inpath = sys.argv[1]

    infiles = glob( pjoin(inpath, 'ws*.root') )

    outtag = re.findall('202\d.*', inpath)[0].replace('/', '')

    # Output ROOT file
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outf = r.TFile(pjoin(outdir, 'delta_htmiss.root'), 'RECREATE')
    
    h_delta_htmiss = r.TH1F('dhtmiss', r'$\Delta H_T^{miss} / H_T^{miss}$', 50, 0, 1)

    save_delta_htmiss(infiles, outtag, outf, h_delta_htmiss)

if __name__ == '__main__':
    main()