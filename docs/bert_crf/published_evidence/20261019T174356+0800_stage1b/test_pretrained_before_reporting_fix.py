"""Opt-in, real pretrained weights + GPU; samples and mapping are synthetic only."""
from datetime import datetime
import gc
import json
from pathlib import Path
import time

import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from bert_crf.config import ModelConfig, configure_runtime
from bert_crf.model import BertEncoder, CharacterBertCRF
from test_crf import legal
from test_model import SyntheticOneToOneAdapter

ROOT = Path(__file__).resolve().parents[2]


def test_real_pretrained_gpu_chain_and_checkpoint(request):
    model_dir = request.config.getoption('--pretrained-dir')
    if not model_dir:
        pytest.skip('Explicit --pretrained-dir required; no implicit downloads')
    output_path = request.config.getoption('--smoke-output')
    checkpoint_dir = request.config.getoption('--checkpoint-dir')
    assert output_path and checkpoint_dir, 'Explicit evidence and ignored checkpoint paths required'
    output_path = Path(output_path)
    assert not output_path.exists(), 'Never overwrite prior smoke evidence'
    assert not Path(checkpoint_dir).exists(), 'Never overwrite prior checkpoint'
    assert torch.cuda.is_available(), 'A real CUDA test cannot pass via CPU fallback'
    config = ModelConfig.load(ROOT / 'configs/bert_crf/stage1b_core_smoke.json')
    device = configure_runtime(config)
    started = time.perf_counter()
    report = {'artifact_kind': config.artifact_kind, 'time': datetime.now().astimezone().isoformat(),
              'synthetic_only': True, 'real_character_adapter': 'NOT_IMPLEMENTED',
              'model_name': config.bert_model_name, 'revision': config.bert_revision,
              'local_model_path': str(Path(model_dir).resolve()),
              'device': torch.cuda.get_device_name(0), 'precision': config.precision,
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    assert tokenizer.is_fast
    # Nonmedical, invented strings; labels below are arbitrary BIO test labels.
    texts = ['天地玄黄', '宇宙']
    batch = tokenizer(texts, return_offsets_mapping=True, return_special_tokens_mask=True,
                      padding=True, truncation=False, return_tensors='pt')
    max_chars = max(map(len, texts))
    mask = torch.arange(max_chars)[None, :] < torch.tensor(list(map(len, texts)))[:, None]
    indices = torch.zeros_like(mask, dtype=torch.long)
    verified_offsets = []
    for b, text in enumerate(texts):
        found = []
        for t, (offset, special, attended) in enumerate(zip(batch['offset_mapping'][b].tolist(),
                batch['special_tokens_mask'][b].tolist(), batch['attention_mask'][b].tolist())):
            if attended and not special:
                found.append((t, tuple(offset)))
        assert [span for _, span in found] == [(i, i + 1) for i in range(len(text))], \
            'Test-only mapping requires verified exactly one token per original character'
        indices[b, :len(text)] = torch.tensor([t for t, _ in found])
        verified_offsets.append(found)
    report.update(synthetic_texts=texts, arbitrary_gold=[[0, 1, 2, 0], [3, 4]],
                  verified_token_offsets=verified_offsets, tokenizer_class=type(tokenizer).__name__,
                  tokenizer_is_fast=tokenizer.is_fast)
    load_start = time.perf_counter()
    encoder = BertEncoder.from_pretrained(config, model_dir)
    report['weight_load_seconds'] = time.perf_counter() - load_start
    assert not encoder.loading_info.get('missing_keys'), encoder.loading_info
    assert not encoder.loading_info.get('mismatched_keys'), encoder.loading_info
    report['loading_info'] = encoder.loading_info
    # Compare real loaded tensors to their safetensors source, not merely shape.
    compared = []
    with safe_open(str(Path(model_dir) / 'model.safetensors'), framework='pt', device='cpu') as source:
        for key in ['embeddings.word_embeddings.weight', 'encoder.layer.11.output.dense.weight']:
            source_key = 'bert.' + key if 'bert.' + key in source.keys() else key
            assert torch.equal(encoder.bert.state_dict()[key], source.get_tensor(source_key))
            compared.append(source_key)
    report['pretrained_tensor_equality_verified'] = compared
    report['encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
    model = CharacterBertCRF(config, encoder, SyntheticOneToOneAdapter()).to(device)
    del encoder
    report['head_parameters'] = sum(p.numel() for p in model.head.parameters())
    report['total_parameters'] = sum(p.numel() for p in model.parameters())
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inputs = dict(input_ids=batch['input_ids'].to(device),
                  attention_mask=batch['attention_mask'].to(device),
                  token_type_ids=batch.get('token_type_ids').to(device),
                  character_mask=mask.to(device),
                  character_labels=torch.tensor([[0, 1, 2, 0], [3, 4, 0, 0]], device=device),
                  mapping={'token_indices': indices.to(device)}, decode=True)
    hidden_shapes = []
    hook = model.encoder.register_forward_hook(lambda module, args, result: hidden_shapes.append(list(result.shape)))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report['baseline_allocated_bytes'] = torch.cuda.memory_allocated()
    model.train()
    t = time.perf_counter()
    result = model(**inputs)
    torch.cuda.synchronize()
    report['forward_seconds_including_decode_validation'] = time.perf_counter() - t
    hook.remove()
    assert hidden_shapes == [[2, batch['input_ids'].shape[1], config.hidden_size]]
    assert result['emissions'].shape == (2, 4, 11)
    assert torch.isfinite(result['loss']) and torch.isfinite(result['emissions']).all()
    assert [len(p) for p in result['paths']] == [4, 2]
    assert all(legal(p) for p in result['paths'])
    report.update(token_hidden_shapes=hidden_shapes, character_emissions_shape=list(result['emissions'].shape),
                  synthetic_smoke_nll=float(result['loss'].detach()), decoded_paths=result['paths'])
    t = time.perf_counter()
    result['loss'].backward()
    torch.cuda.synchronize()
    report['backward_seconds'] = time.perf_counter() - t
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f'Missing gradient: {name}'
        assert torch.isfinite(parameter.grad).all(), f'Nonfinite gradient: {name}'
        gradients[name] = float(parameter.grad.norm())
    for name in ['encoder.bert.embeddings.word_embeddings.weight',
                 'encoder.bert.encoder.layer.0.attention.self.query.weight',
                 'encoder.bert.encoder.layer.11.output.dense.weight',
                 'head.classifier.weight', 'head.crf.transitions',
                 'head.crf.start_transitions', 'head.crf.end_transitions']:
        assert gradients[name] > 0, f'No nonzero gradient: {name}'
    report.update(gradient_tensor_count=len(gradients), all_gradients_finite=True,
                  representative_gradient_norms={n: v for n, v in gradients.items()
                      if n in ['encoder.bert.embeddings.word_embeddings.weight',
                               'encoder.bert.encoder.layer.0.attention.self.query.weight',
                               'encoder.bert.encoder.layer.11.output.dense.weight',
                               'head.classifier.weight', 'head.crf.transitions']},
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del result
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        before = model(**inputs)
        before_emissions = before['emissions'].cpu().clone()
        before_loss = before['loss'].cpu().clone()
        before_paths = before['paths']
    t = time.perf_counter()
    model.save_pretrained(checkpoint_dir)
    report['checkpoint_save_seconds'] = time.perf_counter() - t
    del before, model
    gc.collect()
    torch.cuda.empty_cache()
    t = time.perf_counter()
    reloaded = CharacterBertCRF.from_pretrained(checkpoint_dir, adapter=SyntheticOneToOneAdapter()).to(device).eval()
    assert reloaded.config == config
    torch.cuda.synchronize()
    report['checkpoint_reload_seconds'] = time.perf_counter() - t
    with torch.no_grad():
        after = reloaded(**inputs)
    assert torch.equal(before_emissions, after['emissions'].cpu())
    assert torch.equal(before_loss, after['loss'].cpu())
    assert before_paths == after['paths']
    report.update(checkpoint_path=str(Path(checkpoint_dir).resolve()),
                  checkpoint_emissions_bitwise_equal=True, checkpoint_loss_bitwise_equal=True,
                  checkpoint_paths_equal=True, checkpoint_configuration_equal=True,
                  elapsed_seconds=time.perf_counter() - started, status='PASS', optimizer_steps=0)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
