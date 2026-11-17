"""Persist synthetic-only Day04 checks in a new directory."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.dont_write_bytecode=True
import test_coverage_statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent):raise ValueError('Day04 only')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'run.log').open('x',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_coverage_statistics))
    code=0 if result.wasSuccessful() else 1
    info={'scope':'synthetic_unit_only_not_LLVIP_results','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'exit_code':code,'command':subprocess.list2cmdline(sys.orig_argv),'model_forward_performed':False,'checkpoint_loaded':False}
    with (args.output/'verification.json').open('x',encoding='utf-8') as f:json.dump(info,f,indent=2)
    print((args.output/'run.log').read_text(encoding='utf-8'));return code
if __name__=='__main__':sys.exit(main())
