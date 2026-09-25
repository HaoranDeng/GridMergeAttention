import hashlib
import json
import os
import errno
import time
from pathlib import Path

ROOT=Path(os.environ.get('MONARCH121_ROOT','/local2/denghaoran/GridMergeAttention/experiments/monarchrt_t2v121_20260925'))
def read(path):
    transient={errno.EEXIST,errno.ENOENT,errno.ESTALE,errno.EINTR,errno.EAGAIN,errno.EBUSY,errno.ETIMEDOUT}
    for attempt in range(6):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as exc:
            if isinstance(exc,OSError) and exc.errno not in transient:raise
            if attempt==5:raise
            time.sleep(min(.1*2**attempt,1.))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(temp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
