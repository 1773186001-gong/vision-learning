import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.dont_write_bytecode=True
import test_layer_support

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent):raise ValueError('Day06 only')
    args.output.mkdir(exist_ok=False,parents=True)
    with (args.output/'run.log').open('x',encoding='utf-8') as f:
        r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_layer_support))
    code=0 if r.wasSuccessful() else 1
    with (args.output/'verification.json').open('x',encoding='utf-8') as f:
        json.dump({'scope':'synthetic_only','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),
            'exit_code':code,'command':subprocess.list2cmdline(sys.orig_argv),'model_forward_performed':False,'NMS_executed':False},f,indent=2)
    print((args.output/'run.log').read_text(encoding='utf-8'));return code
if __name__=='__main__':sys.exit(main())
