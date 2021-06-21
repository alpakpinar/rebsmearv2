#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from klepto.archives import dir_archive
from pprint import pprint
from collections import Counter
from tabulate import tabulate

pjoin = os.path.join

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    cfname = 'cutflow_sr_vbf'
    acc.load(cfname)

    cf = acc[cfname]

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    datasets = list(cf.keys())
    cuts = cf[datasets[0]].keys()

    combined_cf = Counter({ cut: 0 for cut in cuts })

    for d in datasets:
        cutflow = Counter(cf[d])
        combined_cf += cutflow

    pcutflow = []
    for idx, (c, v) in enumerate(combined_cf.items()):
        if idx == 0:
            acc = 100
        else:
            acc = v / list(combined_cf.values())[idx-1] * 100
        pcutflow.append([c,v,acc])

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, 'cutflow.txt')
    with open(outpath, 'w+') as f:
        f.write(outtag)
        f.write('\n')

        f.write(
            tabulate(pcutflow, headers=['Cut', 'Number of Events', 'Acceptance (%)'], floatfmt=[".0f", ".0f", ".3f"])
        )

    print(f'File saved: {outpath}')

if __name__ == '__main__':
    main()