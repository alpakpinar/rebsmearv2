#!/usr/bin/env python
import os
import pickle
import subprocess
import htcondor

pjoin = os.path.join

def condor_submit(jobfile):
    cmd = ["bash","/usr/local/bin/condor_submit", jobfile]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Condor submission failed. Stderr:\n {stderr}.")
    jobid = stdout.split()[-1].decode('utf-8').replace('.','')

    return jobid