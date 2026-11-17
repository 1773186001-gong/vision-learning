"""Immutable per-model/image receipts, atomic raw commit and conservative resume."""
import hashlib
import json
import os
from pathlib import Path
import torch

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def json_write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def atomic_torch(path,data):
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('xb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    # Windows rename fails when destination exists; never replace historical/raw files.
    os.rename(tmp,path)
    return sha(path)
class UnknownExecution(RuntimeError):pass
class Ledger:
    def __init__(self,path,signature):self.path=Path(path);self.signature=signature
    def action(self):
        start=self.path/'started.json';raw=self.path/'raw_capture.pth';verified=self.path/'verified.json'
        if not self.path.exists():return 'new'
        if not start.exists():raise UnknownExecution('Incomplete ledger directory: '+str(self.path))
        info=json.loads(start.read_text(encoding='utf-8'))
        if info['signature']!=self.signature:raise RuntimeError('Signature mismatch; never rerun')
        if verified.exists():
            receipt=json.loads(verified.read_text(encoding='utf-8'))
            if not raw.exists() or receipt['raw_sha256']!=sha(raw):raise RuntimeError('Accepted capture checksum differs')
            return 'skip'
        if raw.exists():return 'recover_offline'
        raise UnknownExecution('Started without complete capture; execution UNKNOWN: '+str(self.path))
    def start(self,info):
        if self.action()!='new':raise RuntimeError('Repeated forward forbidden')
        self.path.mkdir(parents=True,exist_ok=False)
        json_write(self.path/'started.json',dict(info,signature=self.signature))
    def commit(self,data):
        if data.get('signature')!=self.signature:raise RuntimeError('Raw signature mismatch')
        h=atomic_torch(self.path/'raw_capture.pth',data)
        json_write(self.path/'captured.json',{'signature':self.signature,'raw_sha256':h})
        return h
    def accept(self,verification):
        if self.action()!='recover_offline':raise RuntimeError('Expected unverified complete raw capture')
        json_write(self.path/'verified.json',dict(verification,signature=self.signature,raw_sha256=sha(self.path/'raw_capture.pth')))
