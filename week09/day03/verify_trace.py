"""Run synthetic unit verification only. Never loads a model, checkpoint or image."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s);stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:stream.flush()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    baseline_path=HERE/"checks/local_unit_protected_before.json"
    before=json.loads(baseline_path.read_text(encoding="utf-8"))
    command=subprocess.list2cmdline(sys.orig_argv)
    with (out/"run.log").open("x",encoding="utf-8") as f:
        stream=Tee(sys.stdout,f)
        stream.write("SYNTHETIC UNIT VERIFICATION ONLY; NOT LLVIP RESULTS\n")
        stream.write("COMMAND "+command+"\n")
        try:
            import torch
            import torchvision
            stream.write(f"ENV python={sys.version} torch={torch.__version__} torchvision={torchvision.__version__}\n")
            suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_trace_rpn_stages.py")
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            changed=[path for path,h in before.items() if not Path(path).is_file() or digest(path)!=h]
            if changed:
                stream.write("PROTECTED INPUT CHANGED: "+repr(changed)+"\n")
            code=0 if result.wasSuccessful() and not changed and result.testsRun>0 else 1
            info=dict(status="passed" if code==0 else "failed",exit_code=code,
                      scope="synthetic_tensor_unit_verification_only",tests_run=result.testsRun,
                      failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                      protected_files_checked=len(before),changed_protected_files=changed,
                      actual_command=command,python=sys.version,platform=platform.platform(),
                      torch=torch.__version__,torchvision=torchvision.__version__,
                      real_model_loaded=False,checkpoint_loaded=False,real_image_forward=False,
                      server_connected=False,
                      source_sha256={x.name:digest(x) for x in
                          (HERE/"trace_rpn_stages.py",HERE/"test_trace_rpn_stages.py",Path(__file__))},
                      source_rpn_sha256=digest(Path(torchvision.__file__).parent/"models/detection/rpn.py"),
                      recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            with (out/"verification.json").open("x",encoding="utf-8") as g:
                json.dump(info,g,ensure_ascii=False,indent=2)
            stream.write(f"UNIT_VERIFICATION_COMPLETE exit_code={code} protected_files={len(before)} changed={len(changed)}\n")
        except BaseException:
            import traceback
            stream.write(traceback.format_exc())
            code=1
        with (out/"execution_status.json").open("x",encoding="utf-8") as g:
            json.dump(dict(exit_code=code,scope="synthetic_unit_only"),g,indent=2)
    return code

if __name__=="__main__":sys.exit(main())

