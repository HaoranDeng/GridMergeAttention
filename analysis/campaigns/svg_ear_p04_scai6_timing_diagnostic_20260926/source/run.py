"""Exclusive two-H200 crossover with identical per-call instrumentation."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from common import ROOT,read,save,sha
from gpu_checks import inventory,clean,reservations


def telemetry():
    fields='index,uuid,name,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit'
    rows=csv.reader(subprocess.check_output(['nvidia-smi','-i','2,4','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).splitlines())
    keys=['index','uuid','model','memory_mib','utilization','sm_clock_mhz','memory_clock_mhz','temperature_c','power_w','power_limit_w']
    result=[]
    for row in rows:
        d={k:v.strip() for k,v in zip(keys,row)}
        for k in keys:
            if k not in ('uuid','model'):d[k]=float(d[k])
        result.append(d)
    return result



if __name__ == '__main__':
    raise SystemExit('Retired crossover: p=.5 was canceled. Telemetry import only.')
