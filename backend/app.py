from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
import re
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, StrictStr, ConfigDict

from .common import ROOT, METHODS, ENTITY_TYPES, config, load, ref, sha, verify_assets
from .text_service import infer, verified_case, shutdown

@asynccontextmanager
async def lifespan(app):
    yield
    shutdown()

app=FastAPI(title='NLPExp 本地实验演示',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])

@app.middleware('http')
async def local_origin(request:Request,call_next):
    origin=request.headers.get('origin')
    if request.method=='POST' and origin and origin not in ['http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8000']:
        return JSONResponse(status_code=403,content={'detail':'仅允许本地演示页面调用'})
    return await call_next(request)

@app.get('/api/health')
def health():return {'status':'ok','training_endpoint':False,'full_dev_http_endpoint':False}

@lru_cache(maxsize=1)
def historical():
    source=ROOT/'backend/resources/history/catalog.json'
    rows=load(source)['runs'];result=[]
    by_method={row['method']:row for row in rows}
    if set(by_method)!=set(METHODS) or len(rows)!=len(METHODS):
        raise ValueError('历史成绩目录必须包含三条唯一技术路线')
    for method in METHODS:
        row=by_method[method]
        evaluation=(ROOT/row['evaluation']['path']).resolve()
        if not evaluation.is_relative_to(source.parent.resolve()):
            raise ValueError('历史评分必须位于登记的资源目录')
        if sha(evaluation)!=row['evaluation']['sha256']:
            raise ValueError('历史评分文件身份不匹配')
        ev=load(evaluation)
        if row['metrics']!=ev['entities']['micro'] or row['counts']!=ev['counts']:
            raise ValueError('历史摘要与原评分不一致')
        result.append({'method':method,'name':METHODS[method],'metrics':row['metrics'],'per_type':ev['entities']['per_type'],
                       'split':'official_dev','evidence_type':'historical_prediction_rescored_in_stage1','counts':row['counts'],
                       'source':ref(source),'evaluation':{**row['evaluation'],**ref(evaluation)},'prediction':row['prediction'],
                       'legacy_metadata_status':row['metadata_status']})
    return result

@app.get('/api/project')
def project():
    return {'name':'NLPExp','subtitle':'IMCS-V2-NER 医疗命名实体识别','task':'字符级 BIO · 严格实体匹配',
            'entity_types':ENTITY_TYPES,'historical':historical(),'counts':historical()[0]['counts'],
            'policy':'strict-ner-character-v1 · 类型与[start,end)边界完全匹配 · 百分比差值用pp',
            'test_notice':'官方test没有公开金标，本应用不提供test F1。',
            'cli':'.\\.venv-bert-crf\\Scripts\\python.exe -B -X utf8 scripts/run_dev_demo.py --split dev --methods rule,softmax,crf'}

resource_cache={}
@app.get('/api/resources')
def resources():
    if resource_cache and time.monotonic()-resource_cache['at']<15:return resource_cache['value']
    states=[]
    for method in METHODS:
        try:states.append({'method':method,'name':METHODS[method],'status':'verified','files':verify_assets(method),
                           'scope':'本次资源文件与冻结身份检查；完整执行状态请查看Dev运行记录'})
        except Exception as e:states.append({'method':method,'name':METHODS[method],'status':'missing_or_mismatch','error':str(e)})
    resource_cache.update(at=time.monotonic(),value=states);return states

def run_dir(run_id):
    if not re.fullmatch(r'dev_demo_[0-9]{8}_[0-9]{6}_[a-f0-9]{8}',run_id):raise HTTPException(404,'未知运行ID')
    root=Path(config()['demo_runs']).resolve();p=(root/run_id).resolve()
    if not p.is_relative_to(root) or not (p/'manifest.json').is_file():raise HTTPException(404,'运行记录不存在')
    return p

@app.get('/api/runs')
def runs():
    root=Path(config()['demo_runs']);found=[]
    if root.exists():
        for p in root.glob('dev_demo_*/manifest.json'):
            try:
                m=load(p);found.append({k:m.get(k) for k in ['run_id','status','started_at','ended_at','elapsed_seconds','split']})
            except (ValueError,OSError):continue
    return sorted(found,key=lambda r:r['started_at'],reverse=True)

@app.get('/api/runs/{run_id}')
def run(run_id:str):
    directory=run_dir(run_id);m=load(directory/'manifest.json')
    if (directory/'post_run_audit.json').is_file():m['post_run_audit']=load(directory/'post_run_audit.json')
    return m

@app.get('/api/runs/{run_id}/logs')
def logs(run_id:str):
    p=run_dir(run_id)/'run.log'
    if not p.exists():return {'text':'','bytes':0}
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-60000));text=f.read().decode('utf-8',errors='replace')
    return {'text':text,'bytes':p.stat().st_size}

def artifact(run_id,key):
    directory=run_dir(run_id);m=load(directory/'manifest.json')
    registered={'manifest':{'path':str(directory/'manifest.json')},'log':{'path':str(directory/'run.log')},
                **{k:v for k,v in m.get('artifacts',{}).items() if v}}
    if (directory/'post_run_audit.json').is_file():registered['post_run_audit']={'path':str(directory/'post_run_audit.json')}
    for method,state in m['methods'].items():
        for k,v in state.get('artifacts',{}).items():
            if v:registered[method+'.'+k]=v
    if key not in registered:raise HTTPException(404,'未登记的证据文件')
    p=Path(registered[key]['path']).resolve()
    if not p.is_relative_to(directory) or not p.is_file():raise HTTPException(404,'证据路径无效')
    return p

@app.get('/api/runs/{run_id}/artifacts/{key}')
def get_artifact(run_id:str,key:str):
    p=artifact(run_id,key)
    return FileResponse(p,media_type='application/json' if p.suffix=='.json' else 'text/plain',filename=p.name)

@app.get('/api/runs/{run_id}/preview/{key}')
def preview(run_id:str,key:str):
    p=artifact(run_id,key)
    with p.open('rb') as f:text=f.read(32000).decode('utf-8',errors='replace')
    return {'file':ref(p),'text':text,'truncated':p.stat().st_size>32000}

@app.get('/api/cases')
def cases():
    case=verified_case()
    return [case] if case else []

@app.get('/api/experiments')
def experiments():
    p=ROOT/'docs/bert_crf/analysis/four_runs_comparison.json';d=load(p)
    return {'split':'internal_val','source':ref(p),'best_models':d['best_models'],'epochs':d['epochs'],
            'configurations':d['configuration_identities'],'error_analysis':d['errors_by_model'],
            'official_dev':historical(),'figures':['four_runs_loss_comparison.png','four_runs_f1_comparison.png','four_runs_error_comparison.png','four_runs_entity_f1_comparison.png'],
            'notice':'内部val用于选模，不能代替官方dev成绩。第四轮边界辅助与最终第三轮E5分别标识。'}

@app.get('/api/figures/{name}')
def figure(name:str):
    if name not in experiments()['figures']:raise HTTPException(404,'未知图表')
    return FileResponse(ROOT/'docs/bert_crf/figures'/name)

class TextInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:StrictStr

@app.post('/api/text/infer')
def text_infer(body:TextInput):
    try:return infer(body.text)
    except ValueError as exc:raise HTTPException(422,str(exc))
    except RuntimeError as exc:raise HTTPException(409,str(exc))

@app.get('/api/text/runs')
def text_runs():
    root=Path(config()['text_runs']);result=[]
    if root.exists():
        for p in sorted(root.glob('text_*/result.json'),reverse=True)[:50]:
            try:
                d=load(p);result.append({k:d.get(k) for k in ['inference_id','started_at','ended_at','status','character_count']})
            except (ValueError,OSError):continue
    return result
