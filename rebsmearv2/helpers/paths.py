#!/usr/bin/env python

def xrootd_format(fpath):
    """Ensure that the file path is file:/* or xrootd"""
    if fpath.startswith("/store/"):
        return f"root://cms-xrd-global.cern.ch//{fpath}"
    elif fpath.startswith("file:") or fpath.startswith("root:"):
        return fpath
    else:
        return f"file://{fpath}"