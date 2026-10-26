import atexit
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
from pathlib import Path
import subprocess
import threading
import time

from .common import ROOT, RUNTIME, METHODS, config, env, gpu_lock, load, now, ref, save, uid
from .entities import entities, strict_score, differences

class Worker:
    def __init__(self, method):
        self.method=method; self.process=None; self.lock=threading.Lock()
        self.executor=ThreadPoolExecutor(max_workers=1)

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process=None

    def predict(self,text):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                RUNTIME.mkdir(parents=True,exist_ok=True)
                log=(RUNTIME/(self.method+'_worker.log')).open('a',encoding='utf-8')
                try:
                    self.process=subprocess.Popen([config()['ml_python'],'-B','-u','-X','utf8','-m','backend.worker','--method',self.method],
                        cwd=ROOT,env=env(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
                finally:log.close()
            self.process.stdin.write(json.dumps({'text':text},ensure_ascii=False)+'\n');self.process.stdin.flush()
            future=self.executor.submit(self.process.stdout.readline)
            try:line=future.result(timeout=180)
            except FutureTimeout:
                self.close();raise RuntimeError('模型推理超时，进程已终止；未返回替代预测')
            if not line:
                self.close();raise RuntimeError('模型进程异常退出，请查看本地worker日志')
            return json.loads(line)

workers={m:Worker(m) for m in METHODS}
def shutdown():
    for w in workers.values():w.close()
atexit.register(shutdown)
gate=threading.Lock()

def verified_case():
    c=load(RUNTIME/'verified_case.json')
    c['gold_entities']=entities(c['raw_text'],c['gold_tags'])
    for h in c['historical'].values():h['entities']=entities(c['raw_text'],h['tags'])
    return c

def infer(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('请输入非空文本')
    if len(text)>500:raise ValueError('最多500个原始Unicode字符；BERT另有冻结token上限，超限会明确拒绝')
    if not gate.acquire(blocking=False):raise RuntimeError('另一个短文本任务正在运行，请稍后重试')
    started=now();run_id=uid('text');directory=Path(config()['text_runs'])/run_id
    directory.mkdir(parents=True,exist_ok=False)
    result={'inference_id':run_id,'kind':'new_custom_text_inference','started_at':started,'raw_text':text,
            'character_count':len(text),'results':[],'gold':None,'status':'running','local_only':True}
    save(directory/'result.json',result)
    try:
        case=verified_case()
        # A gold record is only applicable to the exact original string.
        if text==case['raw_text']:result['gold']={'case_id':case['id'],'tags':case['gold_tags'],'entities':case['gold_entities'],'source':case['source']}
        with gpu_lock():
            for method,worker in workers.items():
                tick=time.perf_counter()
                try:
                    r=worker.predict(text)
                    if r['status']=='success':
                        r['entities']=entities(text,r['character_tags']);r['bio_valid']=True
                        r['raw_text']=text
                        r['score']=strict_score(result['gold']['tags'],r['character_tags']) if result['gold'] else None
                    r['log']=['调用真实冻结资源；输入不含Gold；无训练、无反向传播。'] if r['status']=='success' else ['本路线失败，未返回替代预测。']
                except Exception as exc:r={'method':method,'status':'failed','error':str(exc),'duration_seconds':time.perf_counter()-tick}
                result['results'].append(r);save(directory/'result.json',result)
        result['differences']=differences(result['results'])
        result['status']='completed' if all(r['status']=='success' for r in result['results']) else 'partial'
        result['ended_at']=now();save(directory/'result.json',result)
        save(directory/'file_hashes.json',{'result':ref(directory/'result.json'),'input_characters':len(text),'local_only':True})
        return result
    finally:gate.release()
