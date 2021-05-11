#!/usr/bin/env python

import os
import rebsmearv2

pjoin = os.path.join

def rebsmear_path(path_in_repo):
    return pjoin(rebsmearv2.__path__._path[0], path_in_repo)

def xrootd_format(fpath):
    """Ensure that the file path is file:/* or xrootd"""
    if fpath.startswith("/store/"):
        return f"root://cms-xrd-global.cern.ch//{fpath}"
    elif fpath.startswith("file:") or fpath.startswith("root:"):
        return fpath
    else:
        return f"file://{fpath}"