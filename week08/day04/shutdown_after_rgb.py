"""用户明确授权后的关机守候器：成功完成训练且本地取回核验确认后才关机。"""
import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def ready(job, output):
    state=read(job/'state.json')
    if state['status']=='FAILED': raise RuntimeError('训练失败，保留现场，不自动关机')
    marker=job/'ready_to_shutdown.json'
    if state['status']!='SUCCEEDED' or state['exit_code']!=0 or not marker.is_file(): return False
    confirmation=read(marker)
    completed=read(output/'completed.json')
    assert confirmation['job_name']==job.name and confirmation['local_review_status']=='passed'
    assert confirmation['results_downloaded_and_verified'] is True
    assert completed['epochs']==10 and completed['modality']=='visible'
    assert read(output/'exit_status.json')['exit_code']==0
    assert sha(output/'completed.json')==confirmation['completed_sha256']
    assert sha(output/'best.pth')==confirmation['best_sha256']
    assert sha(output/'last.pth')==confirmation['last_sha256']
    assert read(output/'test_started.json')['checkpoint_sha256']==confirmation['best_sha256']
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    root=Path('D:/vision-learning-server/week08/day04').resolve()
    job=a.job.resolve()
    if socket.gethostname().upper()!='DESKTOP-BFIO0RA' or job.parent!=root/'jobs':
        raise ValueError('不是已授权服务器或Day04任务路径')
    settings=read(job/'parameters.json')
    output=Path(settings['output']).resolve()
    if output.parent!=root/'outputs': raise ValueError('输出路径越界')
    status_path=job/'shutdown_watcher_status.json'
    with status_path.open('x',encoding='utf-8') as stream:
        json.dump({'status':'waiting_for_training_and_verified_download','job':job.name},stream)
    try:
        deadline=time.monotonic()+7200
        while time.monotonic()<deadline:
            try:
                if ready(job,output):
                    status_path.write_text(json.dumps({'status':'verified_waiting_30_seconds_before_shutdown'}),encoding='utf-8')
                    time.sleep(30)
                    # 不加/f；避免/t为正时Windows隐含的强制关闭应用行为。
                    result=subprocess.run(['shutdown.exe','/s','/t','0'],capture_output=True,text=True)
                    status={'status':'shutdown_requested' if result.returncode==0 else 'shutdown_command_failed',
                            'delay_seconds':0,'returncode':result.returncode,
                            'stdout':result.stdout,'stderr':result.stderr}
                    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
                    return result.returncode
            except (FileNotFoundError,json.JSONDecodeError):
                pass  # 只等待尚未写完的状态文件，不重试训练或关机命令。
            time.sleep(5)
        raise TimeoutError('两小时内未收到成功和取回核验确认，未安排关机')
    except BaseException as error:
        status_path.write_text(json.dumps({'status':'stopped_without_shutdown','error':str(error)},
                                         ensure_ascii=False,indent=2),encoding='utf-8')
        raise


if __name__=='__main__': raise SystemExit(main())
