"""Terminal-owned full official-dev inference. No HTTP entry point calls this."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, METHODS, config, env, git_identity, gpu_lock, load, now, ref, save, sha, uid, verify_assets

class Runner:
    def __init__(self, methods):
        self.c = config(); self.run_id = uid('dev_demo')
        self.directory = Path(self.c['demo_runs']) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.m = {'schema_version': 'dev_demo_v1', 'run_id': self.run_id, 'kind': 'full_official_dev_new_inference',
                  'status': 'running', 'started_at': now(), 'ended_at': None, 'runner_pid': os.getpid(),
                  'git': git_identity(), 'split': 'official_dev', 'data': ref(self.c['dev']),
                  'methods': {x: {'method': x, 'name': METHODS[x], 'status': 'queued', 'stage': '等待推理',
                                 'metrics': None, 'exit_codes': {}, 'artifacts': {}} for x in methods},
                  'logs': 'run.log', 'training_executed': False, 'artifacts': {}, 'frontend': 'http://127.0.0.1:5173'}
        self.persist()

    def persist(self): save(self.directory / 'manifest.json', self.m)

    def log(self, message):
        line = now() + ' ' + message
        print(line, flush=True)
        with (self.directory / 'run.log').open('a', encoding='utf-8') as f: f.write(line + '\n')

    def execute(self, args, stage, cwd=ROOT):
        self.log('执行: ' + subprocess.list2cmdline([str(a) for a in args]))
        started = now(); tick = time.perf_counter(); logpath = self.directory / (stage + '.log')
        p = subprocess.Popen([str(a) for a in args], cwd=cwd, env=env(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            with logpath.open('x', encoding='utf-8') as f:
                for line in p.stdout:
                    f.write(line); f.flush(); self.log('[' + stage + '] ' + line.rstrip())
            code = p.wait()
        except BaseException:
            p.terminate(); p.wait(timeout=15); raise
        record = {'argv': [str(a) for a in args], 'cwd': str(cwd), 'started_at': started, 'ended_at': now(),
                  'elapsed_seconds': time.perf_counter() - tick, 'exit_code': code, 'log': ref(logpath),
                  'environment_overrides': {k: env()[k] for k in ['IMCS_DATA_DIR', 'BERT_MODEL_DIR', 'HF_HUB_OFFLINE', 'PYTHONDONTWRITEBYTECODE']}}
        save(self.directory / (stage + '.command.json'), record)
        self.log(f'{stage} 退出码={code}，耗时={record["elapsed_seconds"]:.3f}s')
        return record

    def source_snapshot(self):
        sources = [p for base in [ROOT/'backend', ROOT/'code/src', ROOT/'baseline', ROOT/'evaluation/src',
                                  Path(self.c['crf_reproduce'])] for p in base.rglob('*.py')]
        save(self.directory / 'source_snapshot.json', {'git': self.m['git'], 'files': [ref(p) for p in sorted(set(sources))]})
        return ref(self.directory / 'source_snapshot.json')

    def metadata(self, method, record, pred, assets):
        c = self.c; d = self.directory / method
        inline = {'method': method, 'split': 'dev', 'batch_size': 64 if method == 'softmax' else 16,
                  'min_count': 10 if method == 'rule' else None, 'assets': assets,
                  'no_training': True, 'data': self.m['data']}
        save(d / 'runtime_config.json', inline)
        info = {'schema_version': '1.0', 'artifact_kind': 'model_prediction', 'run_id': self.run_id+'_'+method,
                'method': {'name': METHODS[method], 'kind': 'rule' if method == 'rule' else 'bert'},
                'method_version': 'frozen:' + assets[0]['sha256'],
                'data': {'version': 'IMCS-V2-NER platform archive v2', 'split': 'dev', 'source_file': c['dev'],
                         'source_sha256': self.m['data']['sha256'], 'archive_sha256': sha(ROOT/'IMCS-V2-NER.zip')},
                'code': {**self.m['git'], 'snapshot': self.m['source_snapshot']},
                'config': {'version': 'demo_inference_adapter_v1', **ref(d/'runtime_config.json')},
                'preprocessing': '原始Unicode字符不归一化；输入投影为原对话ID、句ID、文本；Gold不输入模型。',
                'alignment': {'tokenizer': '冻结Tokenizer，资产逐文件SHA记录' if method != 'rule' else '无分词器，原字符匹配',
                              'character_mapping': '复用原词片offset恢复' if method == 'softmax' else '冻结字符映射/CRF' if method == 'crf' else '原始字符位置',
                              'special_tokens': '特殊token与Padding不输出', 'long_text': '超出冻结长度或覆盖失败即报错，不截断、不补O'},
                'prediction': {'file': str(pred), 'sha256': sha(pred)}, 'logs': [record['log']],
                'inference_command': subprocess.list2cmdline(record['argv']), 'inference_working_directory': record['cwd'],
                'inference_elapsed_seconds': record['elapsed_seconds'],
                'timing_scope': '外层真实子进程墙钟时间，包含导入、资源检查、加载、预测与写盘，不含评分或训练。',
                'device': 'cpu' if method == 'rule' else 'cuda',
                'environment': load(Path(c['evidence'])/'ml_environment.json'),
                'adaptation_notice': 'New run metadata for actual execution; original historical run_info files untouched. Automatic evidence presence checks are not human attestations.'}
        if method == 'rule':
            info.update(training_applicability='本次仅使用已冻结词表推理，不进行词典构建或训练；词表与源码身份已记录，不反推构建时长。',
                        seed_applicability='冻结词典最长匹配，无随机过程', random_seeds=None)
        else:
            train = Path(c['crf_training_run']) if method == 'crf' else ROOT/'runs'/c['softmax_source_run']
            logs = list(train.glob('*.log'))
            if not logs: raise FileNotFoundError('原训练日志不存在')
            cfg = Path(c['crf_reproduce'])/'training_config.json' if method == 'crf' else train/'config.json'
            seed = load(cfg)['model']['seed'] if method == 'crf' else load(cfg)['config']['seed']
            info['random_seeds'] = {'training_seed': seed}
            info['training'] = {'command': subprocess.list2cmdline(load(train/'run_manifest.json')['command']) if method == 'crf' and 'command' in load(train/'run_manifest.json') else
                                 ('详见原训练run_manifest.json；未在本次执行训练' if method == 'crf' else '原组员记录：python -m src.train --run-id '+c['softmax_source_run']+'；实际参数见原config.json'),
                                'model_version': 'E5训练source_snapshot及manifest可追溯；checkpoint:'+assets[0]['sha256'] if method == 'crf' else '组员冻结checkpoint:'+assets[0]['sha256']+'；原训练Git revision未提供',
                                'data': [{**ref(c['original_train']), 'split': 'train'}],
                                'logs': [ref(p) for p in logs], 'config': ref(cfg)}
        save(d/'run_info.json', info)
        return d/'run_info.json'

    def run(self):
        tick = time.perf_counter(); self.log('Run ID: '+self.run_id)
        self.log('仅执行dev推理、校验与评分；前端查看 http://127.0.0.1:5173')
        try:
            if self.m['data']['sha256'] != '16f597ad368971c7db994ca904e05d91942f0e3c73009c9ef1d597fde065d39f':
                raise ValueError('官方dev数据SHA不匹配')
            self.m['source_snapshot'] = self.source_snapshot(); self.persist()
            for method, state in self.m['methods'].items():
                d = self.directory / method; d.mkdir()
                try:
                    state.update(status='running', stage='检查冻结资源', started_at=now()); self.persist()
                    assets = verify_assets(method); state['assets'] = assets
                    save(d/'assets.json', assets); self.log(METHODS[method]+' 资源身份核验通过')
                    pred_dir = d/'prediction'; py = self.c['ml_python']
                    if method == 'rule':
                        args = [py, '-B', '-u', '-X', 'utf8', '-m', 'backend.worker', '--method', 'rule', '--dev', '--output', pred_dir]; cwd=ROOT
                    elif method == 'softmax':
                        args = [py, '-B', '-u', '-X', 'utf8', '-m', 'src.predict', '--run-id', self.c['softmax_source_run'], '--split', 'dev', '--batch-size', '64', '--output-dir', pred_dir]; cwd=ROOT/'code'
                    else:
                        original_cfg = load(self.c['crf_inference_config'])
                        cfg = {**original_cfg, 'run_id': self.run_id+'_crf',
                               'source': self.c['dev'], 'source_sha256': self.m['data']['sha256']}
                        save(d/'path_adaptation.json', {'reason':'本次推理与统一评分使用同一绝对dev路径；数据字节哈希不变。冻结模型、Tokenizer、字符映射和参数不变。',
                                                       'original_config':ref(self.c['crf_inference_config']),
                                                       'changes':{k:{'before':original_cfg.get(k),'after':v} for k,v in cfg.items() if original_cfg.get(k)!=v}})
                        save(d/'frozen_inference_config.json', cfg)
                        state['artifacts']['inference_config']=ref(d/'frozen_inference_config.json')
                        state['artifacts']['path_adaptation']=ref(d/'path_adaptation.json')
                        args = [py, '-B', '-u', '-X', 'utf8', str(Path(self.c['crf_reproduce'])/'infer.py'), '--config', d/'frozen_inference_config.json', '--project-root', ROOT, '--output', pred_dir]; cwd=ROOT
                    state['stage']='模型加载 / 完整dev预测'; self.persist()
                    with gpu_lock() if method != 'rule' else nullcontext():
                        rec = self.execute(args, method+'_inference', cwd)
                    state['exit_codes']['inference'] = rec['exit_code']
                    if rec['exit_code']: raise RuntimeError('真实推理失败，查看日志')
                    pred=pred_dir/'predictions_dev.json'
                    state['artifacts']['prediction']=ref(pred)
                    state['stage']='独立预测校验'; self.persist()
                    val=self.execute([py,'-B','-X','utf8','src/validate_predictions.py','--data',self.c['dev'],'--predictions',pred,'--split','dev','--report',d/'validation.json'],method+'_validation',ROOT/'evaluation')
                    state['exit_codes']['validation']=val['exit_code']; state['artifacts']['validation']=ref(d/'validation.json')
                    if val['exit_code']: raise RuntimeError('完整预测校验失败')
                    self.metadata(method,rec,pred,assets)
                    state['artifacts']['run_info']=ref(d/'run_info.json')
                    state.update(status='predicted', stage='预测与校验完成，等待统一评分')
                except Exception as exc:
                    state.update(status='failed', stage='失败', error=str(exc)); self.log(METHODS[method]+' 失败: '+str(exc))
                self.persist()
            entries=[{'run_id':self.run_id+'_'+m,'method_name':METHODS[m],'predictions':s['artifacts']['prediction']['path'],
                      'run_info':s['artifacts']['run_info']['path']} for m,s in self.m['methods'].items() if s['status']=='predicted']
            if entries:
                save(self.directory/'evaluation_input.json',{'schema_version':'1.0','purpose':'model_evaluation','data':{'path':self.c['dev'],'split':'dev'},'runs':entries})
                for s in self.m['methods'].values():
                    if s['status']=='predicted':s.update(status='scoring',stage='统一评分与元数据核查')
                self.persist()
                rec=self.execute([self.c['ml_python'],'-B','-u','-X','utf8','src/run_evaluation_batch.py','--manifest',self.directory/'evaluation_input.json','--output-dir',self.directory/'evaluation'],'unified_evaluation',ROOT/'evaluation')
                batch_path=next((self.directory/'evaluation').rglob('batch_report.json')); batch=load(batch_path)
                self.m['artifacts']['batch_report']=ref(batch_path)
                self.m['artifacts']['comparison']=ref(batch['comparison']['path'])
                self.m['batch_exit_code']=rec['exit_code'];self.m['batch_status']=batch['status']
                for outcome in batch['runs']:
                    method=outcome['run_id'].rsplit('_',1)[-1];state=self.m['methods'][method]
                    state['exit_codes']['batch']=rec['exit_code']
                    if 'evaluation_path' not in outcome:
                        state.update(status='failed',stage='评分失败',error=outcome.get('reason','未生成评分'));continue
                    ev=load(outcome['evaluation_path'])
                    state.update(status='completed' if outcome['status']=='SCORED' else 'partial',stage='完成' if outcome['status']=='SCORED' else '已评分，证据检查未全通过',
                                 metrics=ev['entities']['micro'],per_type=ev['entities']['per_type'],counts=ev['counts'],
                                 metadata_status=outcome['metadata_status'],automated_metadata_status=outcome['automated_metadata_status'],ended_at=now())
                    state['artifacts']['evaluation']=ref(outcome['evaluation_path'])
                    state['artifacts']['metadata_check']=ref(Path(outcome['evaluation_path']).parent/'reproducibility_check.json')
                    state['artifacts']['error_summary']=ref(Path(outcome['evaluation_path']).parent/'errors/error_summary.json') if (Path(outcome['evaluation_path']).parent/'errors/error_summary.json').exists() else None
                self.m['status']='completed' if rec['exit_code']==0 and all(s['status']=='completed' for s in self.m['methods'].values()) else 'partial'
            else:self.m['status']='failed'
        except KeyboardInterrupt:
            self.m.update(status='interrupted',error='用户中断')
        except Exception as exc:
            self.m.update(status='failed',error=str(exc));self.log('运行失败: '+str(exc))
        finally:
            self.m.update(ended_at=now(),elapsed_seconds=time.perf_counter()-tick); self.persist()
            self.log('整体状态: '+self.m['status'])
            for method,s in self.m['methods'].items():self.log(METHODS[method]+': '+s['status']+' '+str(s.get('metrics')))
            self.log('运行目录: '+str(self.directory));self.log('前端: http://127.0.0.1:5173')
        return 0 if self.m['status']=='completed' else 2 if self.m['status']=='partial' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['dev'],default='dev');p.add_argument('--methods',default='rule,softmax,crf')
    a=p.parse_args();methods=a.methods.split(',')
    if len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):p.error('methods只接受rule,softmax,crf且不重复')
    return Runner(methods).run()

if __name__=='__main__':raise SystemExit(main())
