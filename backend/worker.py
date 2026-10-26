"""Isolated model imports, inference-only adapters, optional JSON-line service."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

from .common import ROOT, config, load, ref, save, verify_assets

class Adapter:
    def __init__(self, method):
        self.method = method
        self.assets = verify_assets(method)
        c = config()
        if method == 'rule':
            sys.path.insert(0, str(ROOT / 'baseline'))
            from run_rule import load_lexicon
            from fast_match import build_index, all_types, select_bio, collect_candidates
            lexicon, conflicts = load_lexicon(min_count=10)
            if len(lexicon) != 570: raise ValueError('冻结mc10词典条目数异常')
            self.index = build_index(lexicon); self.allowed = all_types(self.index)
            self.rule = lambda text: select_bio(text, collect_candidates(text, self.index), self.allowed)
            self.fingerprint = lambda: json.dumps(self.index, sort_keys=True, ensure_ascii=False)
            self.model_id = 'frozen_mc10_570'
        else:
            import torch
            self.torch = torch
            if not torch.cuda.is_available(): raise RuntimeError('冻结演示环境需要可用CUDA')
            if method == 'softmax':
                sys.path.insert(0, str(ROOT / 'code'))
                from src.dataset import load_tokenizer, build_loader
                from src.model import build_model
                from src.metrics import decode_batch
                self.tokenizer = load_tokenizer(); self.loader = build_loader; self.decode = decode_batch
                self.model = build_model().to('cuda')
                checkpoint = torch.load(c['softmax_checkpoint'], map_location='cpu', weights_only=False)
                self.model.load_state_dict(checkpoint['model_state'], strict=True)
                self.model_id = c['softmax_source_run'] + '/epoch_3'
            elif method == 'crf':
                frozen = Path(c['crf_reproduce'])
                spec = importlib.util.spec_from_file_location('e5_infer', frozen / 'infer.py')
                self.frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.frozen)
                from transformers import AutoTokenizer, BertConfig, BertModel
                from bert_crf.config import ModelConfig, configure_runtime
                from bert_crf.training_config import TrainingConfig
                from bert_crf.model import CharacterBertCRF, BertEncoder
                from bert_crf.character_mapping import OffsetCharacterAdapter
                from bert_crf.checkpoint import load_checkpoint_model
                formal = load(frozen / 'training_config.json')
                self.mc = ModelConfig(**formal['model']); tc = TrainingConfig(**formal['training'])
                configure_runtime(self.mc)
                self.tokenizer = AutoTokenizer.from_pretrained(ROOT / formal['inputs']['tokenizer']['path'], use_fast=True, local_files_only=True)
                bc = BertConfig.from_json_file(ROOT / formal['inputs']['pretrained']['path'] / 'config.json')
                self.model = CharacterBertCRF(self.mc, BertEncoder(BertModel(bc, add_pooling_layer=False)), OffsetCharacterAdapter()).to('cuda')
                cfg = load(c['crf_inference_config'])
                state = load_checkpoint_model(c['crf_checkpoint'], self.model, cfg['model_identity'], tc.to_dict())
                assert state['epoch'] == 5 and not any('boundary' in name for name, _ in self.model.named_parameters())
                self.max_tokens = cfg['max_tokens']; self.model_id = 'frozen_e5_' + cfg['model_identity']
            else:
                raise ValueError('unknown method')
            self.model.eval()
            def fingerprint():
                import hashlib
                h = hashlib.sha256()
                for name, t in sorted(self.model.state_dict().items()):
                    h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
                return h.hexdigest()
            self.fingerprint = fingerprint

    def predict(self, text):
        before = self.fingerprint()
        changed = 0
        try:
            if self.method != 'rule': self.model.to('cuda')
            if self.method == 'rule':
                tags = self.rule(text)
            elif self.method == 'softmax':
                examples = [{'dialogue_id': 'local', 'sentence_id': 'input', 'text': text, 'tags': None}]
                loader = self.loader(examples, self.tokenizer, batch_size=1, shuffle=False, with_labels=False)
                with self.torch.inference_mode():
                    for batch in loader:
                        logits = self.model(input_ids=batch['input_ids'].to('cuda'), attention_mask=batch['attention_mask'].to('cuda')).logits
                        decoded, changed = self.decode(logits.argmax(-1).cpu(), batch['offsets'], batch['text'])
                        tags = decoded[0]
            else:
                rows = [('local', 'input', text)]
                encoded = self.frozen.encode(self.tokenizer, rows, self.max_tokens)
                with self.torch.inference_mode():
                    output = self.model(**self.frozen.batch_inputs(self.tokenizer, rows, encoded, 'cuda'), decode=True)
                assert output['loss'] is None
                tags = [self.mc.labels[i] for i in output['paths'][0]]
            if len(tags) != len(text): raise ValueError('原始字符与标签不等长')
        finally:
            # Keep cached weights on CPU between requests; the full-dev CLI owns
            # the same GPU lock and must not compete with idle GPU model caches.
            if self.method != 'rule':
                self.model.to('cpu')
                self.torch.cuda.empty_cache()
            after = self.fingerprint()
            if before != after: raise RuntimeError('推理前后参数发生变化')
            if self.method != 'rule' and any(p.grad is not None for p in self.model.parameters()):
                raise RuntimeError('推理不应产生梯度')
        return {'character_tags': tags, 'model_or_dictionary_id': self.model_id,
                'artifact_hash': self.assets[0]['sha256'], 'asset_files': self.assets,
                'parameter_hash_before': before if self.method != 'rule' else None,
                'parameter_hash_after': after if self.method != 'rule' else None,
                'parameters_unchanged': True, 'backward_calls': 0, 'optimizer_steps': 0,
                'bio_repairs': changed, 'warnings': [f'原Softmax解码流程BIO修复 {changed} 处'] if changed else []}

def serve(method):
    protocol = sys.stdout
    sys.stdout = sys.stderr  # model loading logs never corrupt the JSON response stream
    adapter = None
    for line in sys.stdin:
        started = time.perf_counter()
        try:
            request = json.loads(line)
            if adapter is None: adapter = Adapter(method)
            result = {'status': 'success', **adapter.predict(request['text'])}
        except Exception as exc:
            # No original text is written to general-purpose logs.
            result = {'status': 'failed', 'error': str(exc), 'error_type': type(exc).__name__}
        result.update(method=method, duration_seconds=time.perf_counter() - started)
        protocol.write(json.dumps(result, ensure_ascii=False) + '\n'); protocol.flush()

def rule_dev(output):
    adapter = Adapter('rule'); c = config()
    data = load(c['dev']); predictions = {}; done = 0
    print('冻结mc10已加载；仅读取原始句子文本', flush=True)
    for did, d in data.items():
        predictions[did] = {}
        for row in d['dialogue']:
            tags = adapter.rule(row['sentence'])
            assert len(tags) == len(row['sentence'])
            predictions[did][row['sentence_id']] = ' '.join(tags)
            done += 1
            if done % 5000 == 0: print(f'已处理 {done} 句', flush=True)
    assert len(predictions) == 833 and done == 33267
    Path(output).mkdir(parents=True, exist_ok=False)
    p = Path(output) / 'predictions_dev.json'
    with p.open('x', encoding='utf-8', newline='\n') as f: json.dump(predictions, f, ensure_ascii=False)
    print(json.dumps({'event': 'prediction_written', 'sentences': done, 'prediction': ref(p)}, ensure_ascii=False), flush=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--method', choices=['rule', 'softmax', 'crf'], required=True)
    p.add_argument('--dev', action='store_true'); p.add_argument('--output')
    args = p.parse_args()
    if args.dev:
        if args.method != 'rule' or not args.output: p.error('dev worker is only the frozen rule adapter')
        rule_dev(args.output)
    else:
        serve(args.method)

if __name__ == '__main__': main()
