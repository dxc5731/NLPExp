"""Character trainer: isolated synthetic tests or explicitly preflighted real runs."""
from dataclasses import asdict, replace
import math
from pathlib import Path
import time
from typing import Protocol

import torch

from .checkpoint import load_checkpoint_model, restore_checkpoint, save_checkpoint
from .config import configure_runtime
from .experiment import Experiment, environment, file_hash, fingerprint, write_json
from .metrics import Metrics
from .training_config import TrainingConfig, require_synthetic


class BatchProvider(Protocol):
    """Epoch-pure provider contract (no hidden cursor/worker RNG).

    size: exact number of original sentences; metadata: version/data/split hashes,
    tokenizer and alignment IDs, artifact_kind. batches(epoch,batch_size,shuffle,
    seed) must deterministically cover every sample once, using a dedicated
    epoch-seeded generator, without consuming global model/dropout RNG.

    Each batch: {'ids': [(dialogue_id,sentence_id), ...], 'inputs': model kwargs}.
    inputs must include original character_labels and prefix character_mask.
    Future BERT inputs/mapping are passed through untouched to the existing model.
    Real providers require a matching validated formal execution context.
    """
    size: int
    metadata: dict

    def batches(self, epoch: int, batch_size: int, shuffle: bool, seed: int): ...


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def metric_value(metrics, name):
    value = metrics
    try:
        for part in name.split('.'):
            value = value[part]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Missing selection metric: {name}') from exc
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'Invalid selection metric: {name}')
    return value


class Trainer:
    def __init__(self, model, config: TrainingConfig, train: BatchProvider, val: BatchProvider,
                 output_root, *, metadata=None, run_id=None, execution_context=None,
                 record_validation_artifacts=False):
        self.execution_context = execution_context
        if config.artifact_kind == 'synthetic_smoke_test':
            require_synthetic(config)
            if execution_context is not None:
                raise ValueError('Synthetic training cannot use a real execution context')
        else:
            # Retain the old refusal unless the ENTIRE formal startup succeeded.
            if config.artifact_kind != 'real_training' or execution_context is None:
                raise RuntimeError('Real training disabled without a validated formal preflight context')
            from .preflight import ReadyContext
            if not isinstance(execution_context, ReadyContext):
                raise ValueError('Invalid formal execution context')
            execution_context.validate_training(model, config, train, val)
            expected_root = execution_context.config.path(execution_context.config.value['outputs']['root'])
            if Path(output_root).resolve() != expected_root:
                raise ValueError('Formal output root/config mismatch')
            metadata = {**(metadata or {}), 'formal_config': execution_context.config.value,
                        'preflight': execution_context.report}
            record_validation_artifacts = True
        for provider in (train, val):
            if config.artifact_kind == 'synthetic_smoke_test':
                require_synthetic(config, provider.metadata.get('artifact_kind', 'UNSPECIFIED'))
            if type(provider.size) is not int or provider.size <= 0:
                raise ValueError('Training/validation data must be nonempty')
        self.model, self.config, self.train_provider, self.val_provider = model, config, train, val
        model_config = model.config
        runtime = replace(model_config, seed=config.seed, device=config.device, precision=config.precision,
                          deterministic_algorithms=config.deterministic_algorithms)
        # Reject misleading model/runtime records, rather than silently overriding.
        if (model_config.seed, model_config.device, model_config.precision, model_config.deterministic_algorithms) != \
                (config.seed, config.device, config.precision, config.deterministic_algorithms):
            raise ValueError('Model/training runtime configuration mismatch')
        self.device = configure_runtime(runtime)
        model.to(device=self.device, dtype=torch.float32)
        self.steps_per_epoch = math.ceil(train.size / config.batch_size)
        total_steps = self.steps_per_epoch * config.epochs
        if config.warmup_steps >= total_steps:
            raise ValueError('warmup_steps must be less than total planned optimizer steps')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
            betas=config.adam_betas, eps=config.adam_epsilon, weight_decay=config.weight_decay, foreach=False)

        def schedule(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            return 1.0 if config.scheduler == 'constant' else max(0.0, (total_steps - step) / (total_steps - config.warmup_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.identity = fingerprint({'model_class': type(model).__module__ + '.' + type(model).__qualname__,
            'model_config': asdict(model_config), 'encoder_config': model.encoder.bert.config.to_dict() if hasattr(model, 'encoder') else None,
            'adapter_class': (type(model.adapter).__module__ + '.' + type(model.adapter).__qualname__) if hasattr(model, 'adapter') else None,
            'parameters': [(n, list(p.shape), str(p.dtype), p.requires_grad) for n, p in model.named_parameters()],
            'train_provider': train.metadata, 'val_provider': val.metadata,
            'train_size': train.size, 'val_size': val.size,
            'runtime': environment()['packages'], 'device': config.device})
        combined = {'training': config.to_dict(), 'model': asdict(model_config)}
        if execution_context is not None:
            combined['formal'] = execution_context.config.value
        self.run = Experiment(output_root, config.artifact_kind, combined,
            {'train': train.metadata, 'validation': val.metadata, 'model_identity': self.identity,
             'loader_policy': 'epoch-seeded provider; no persistent sampler cursor; num_workers=0',
             'selection': {'metric': config.selection_metric, 'mode': config.selection_mode,
                           'tie': 'retain earlier best; min_delta is strict', 'patience': config.early_stopping_patience},
             'user_metadata': metadata or {}}, run_id)
        self.state = {'epoch': 0, 'global_step': 0, 'best_value': None, 'best_epoch': None,
                      'bad_epochs': 0, 'best_checkpoint': None, 'best_checkpoint_sha256': None}
        self._fitted = False
        self.record_validation_artifacts = record_validation_artifacts
        self._validation_epoch = None
        self._last_validation_artifact = None
        if record_validation_artifacts:
            from .run_artifacts import initialize_run_info
            initialize_run_info(self)

    def _batch(self, batch):
        if not isinstance(batch, dict) or 'inputs' not in batch or 'ids' not in batch:
            raise ValueError('Batch requires inputs and original IDs')
        inputs = move(batch['inputs'], self.device)
        if 'character_labels' not in inputs or inputs['character_labels'] is None:
            raise ValueError('Missing original character_labels')
        mask = inputs.get('character_mask')
        labels = inputs['character_labels']
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2 or 0 in mask.shape:
            raise ValueError('Invalid character_mask')
        if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError('Nonempty prefix character_mask required')
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != mask.shape:
            raise ValueError('Invalid character_labels shape/dtype')
        if bool(((labels < 0) | (labels >= 11)).any()):
            raise ValueError('Labels must be 0..10, never -100')
        ids = batch['ids']
        if len(ids) != mask.shape[0] or len(set(ids)) != len(ids):
            raise ValueError('ID count mismatch or duplicates')
        if any(not isinstance(k, tuple) or len(k) != 2 or any(not isinstance(x, str) or not x for x in k) for k in ids):
            raise ValueError('Original ID pairs must be nonempty strings')
        return inputs, ids

    @staticmethod
    def _loss(output):
        loss = output.get('loss')
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError('Loss is missing, nonscalar or NaN/Inf')
        return loss

    @torch.no_grad()
    def evaluate(self):
        mode = self.model.training
        self.model.eval()
        metric = Metrics()
        total_loss = count = 0
        tick = time.perf_counter()
        writer = None
        try:
            if self.record_validation_artifacts and self._validation_epoch is not None:
                from .run_artifacts import ValidationArtifacts
                writer = ValidationArtifacts(self.run.path, self._validation_epoch, self.config.artifact_kind)
            for batch in self.val_provider.batches(0, self.config.batch_size, False, self.config.seed):
                inputs, ids = self._batch(batch)
                output = self.model(**inputs, decode=True)
                loss = self._loss(output)
                paths = output.get('paths')
                if not isinstance(paths, list) or len(paths) != len(ids):
                    raise ValueError('Decode missing or batch length mismatch')
                for b, key in enumerate(ids):
                    length = int(inputs['character_mask'][b].sum())
                    gold = inputs['character_labels'][b, :length].tolist()
                    metric.update(key, gold, paths[b])
                    if writer is not None:
                        writer.add(key, gold, paths[b])
                total_loss += float(loss) * len(ids)
                count += len(ids)
            if count != self.val_provider.size:
                raise ValueError('Validation provider sentence count mismatch')
            result = {'loss': total_loss / count, **metric.compute()}
            if writer is not None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                self._last_validation_artifact = writer.finish(result, time.perf_counter() - tick)
            return result
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        finally:
            self.model.train(mode)

    def load_best(self):
        path = self.state['best_checkpoint']
        if not path or file_hash(path) != self.state['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint missing or hash mismatch')
        return load_checkpoint_model(path, self.model, self.identity, self.config.to_dict())

    def fit(self, *, resume_from=None, stop_after_epoch=None):
        """One invocation per run directory; interruptions resume in a NEW run.

        stop_after_epoch is an explicit test/operational boundary, never a change
        to configured total epochs or scheduler horizon. No batch-boundary resume.
        """
        if self._fitted:
            raise RuntimeError('Use a new run directory for each fit/resume invocation')
        self._fitted = True
        started = time.perf_counter()
        self.run.status('running')
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            if resume_from:
                self.state = restore_checkpoint(resume_from, self.model, self.optimizer, self.scheduler,
                                                self.identity, self.config.to_dict())
                if self.state['epoch'] < 1 or self.state['epoch'] > self.config.epochs or self.state['global_step'] != self.state['epoch'] * self.steps_per_epoch:
                    raise ValueError('Invalid epoch-boundary checkpoint counters')
                if self.state['best_checkpoint_sha256'] is None:
                    if Path(self.state['best_checkpoint']).resolve() != Path(resume_from).resolve():
                        raise ValueError('Missing best checkpoint identity')
                    self.state['best_checkpoint_sha256'] = file_hash(resume_from)
                if file_hash(self.state['best_checkpoint']) != self.state['best_checkpoint_sha256']:
                    raise ValueError('Best checkpoint missing or hash mismatch on resume')
                self.run.status('running', resumed_from=str(Path(resume_from).resolve()),
                                resumed_from_sha256=file_hash(resume_from))
                self.run.event('resume', state=self.state)
            if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not self.state['epoch'] < stop_after_epoch <= self.config.epochs):
                raise ValueError('stop_after_epoch must be after checkpoint epoch and within configured epochs')
            final_epoch = stop_after_epoch or self.config.epochs
            early_stop = self.config.early_stopping_patience is not None and self.state['bad_epochs'] >= self.config.early_stopping_patience
            for epoch in range(self.state['epoch'] + 1, final_epoch + 1):
                if early_stop:
                    break
                t = time.perf_counter()
                self.model.train()
                loss_sum, count, seen, steps = 0.0, 0, set(), 0
                for batch in self.train_provider.batches(epoch, self.config.batch_size, self.config.shuffle, self.config.seed):
                    inputs, ids = self._batch(batch)
                    if seen.intersection(ids):
                        raise ValueError('Duplicate training sentence ID in epoch')
                    seen.update(ids)
                    self.optimizer.zero_grad(set_to_none=True)
                    output = self.model(**inputs, decode=False)
                    loss = self._loss(output)
                    loss.backward()
                    grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                    if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                        raise FloatingPointError('Missing or NaN/Inf gradients')
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True)
                    lr_used = self.optimizer.param_groups[0]['lr']
                    self.optimizer.step()
                    self.scheduler.step()
                    self.state['global_step'] += 1
                    count += len(ids)
                    steps += 1
                    loss_sum += float(loss.detach()) * len(ids)
                    self.run.event('train_batch', epoch=epoch, global_step=self.state['global_step'],
                                   samples=len(ids), loss=float(loss.detach()), learning_rate_used=lr_used,
                                   gradient_norm_before_clipping=float(grad_norm))
                if count != self.train_provider.size or steps != self.steps_per_epoch:
                    raise ValueError('Training provider sentence/batch count mismatch')
                self._validation_epoch = epoch
                metrics = self.evaluate()
                self._validation_epoch = None
                value = metric_value(metrics, self.config.selection_metric)
                best = self.state['best_value']
                improved = best is None or ((value - best if self.config.selection_mode == 'max' else best - value) > self.config.min_delta)
                path = (self.run.path / 'checkpoints' / f'epoch_{epoch:04d}.pt').resolve()
                self.state.update(epoch=epoch, bad_epochs=0 if improved else self.state['bad_epochs'] + 1)
                if improved:
                    self.state.update(best_value=value, best_epoch=epoch, best_checkpoint=str(path), best_checkpoint_sha256=None)
                # Hash of this file cannot be embedded in itself. A best's hash
                # is recorded in checkpoint_info; later epochs carry it forward.
                checksum = save_checkpoint(path, self.model, self.optimizer, self.scheduler,
                    self.state, self.identity, self.config.to_dict(),
                    run_metadata={'run_id': self.run.manifest['run_id'], 'config': combined_config(self),
                                  'code': self.run.manifest['code']})
                if improved:
                    self.state['best_checkpoint_sha256'] = checksum
                self.run.event('checkpoint', epoch=epoch, path=str(path), sha256=checksum,
                               improved=improved, metric=self.config.selection_metric, value=value)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                row = {'epoch': epoch, 'train_loss': loss_sum / count, 'val_loss': metrics['loss'],
                       **{f'val_{short}_{m}': metrics[level]['micro'][m] for short, level in [('char', 'character'), ('entity', 'entity')] for m in ('precision', 'recall', 'f1')},
                       'learning_rate': self.optimizer.param_groups[0]['lr'],
                       'epoch_elapsed_seconds': time.perf_counter() - t, 'global_step': self.state['global_step']}
                self.run.epoch(row, metrics)
                if self.record_validation_artifacts:
                    from .run_artifacts import update_run_info
                    update_run_info(self, 'running')
                write_json(self.run.path / 'checkpoint_info.json', {'artifact_kind': self.config.artifact_kind,
                    'latest': str(path), 'latest_sha256': checksum, **self.state,
                    'selection_reason': 'strict improvement' if improved else 'tie/insufficient improvement; earlier best retained'})
                early_stop = self.config.early_stopping_patience is not None and self.state['bad_epochs'] >= self.config.early_stopping_patience
            status = 'early_stopped' if early_stop else ('interrupted' if self.state['epoch'] < self.config.epochs else 'completed')
            self.run.status(status, elapsed_seconds=time.perf_counter() - started, state=self.state,
                timing_scope='fit includes updates, validation, logging and checkpoint I/O; excludes initial model/provider load',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None)
            if self.record_validation_artifacts:
                from .run_artifacts import finalize_artifacts
                finalize_artifacts(self, status)
            return self.state.copy()
        except BaseException as exc:
            self.run.status('failed', exception_type=type(exc).__name__, error=str(exc),
                            elapsed_seconds=time.perf_counter() - started)
            try:
                self.run.event('failure', exception_type=type(exc).__name__, message=str(exc),
                               completed_epoch=self.state['epoch'], global_step=self.state['global_step'])
            except OSError:
                # Manifest still records failure if train.log alone is unwritable.
                pass
            if self.record_validation_artifacts:
                from .run_artifacts import update_run_info
                try:
                    update_run_info(self, 'failed')
                except OSError:
                    pass
            raise


def combined_config(trainer):
    return {'training': trainer.config.to_dict(), 'model': asdict(trainer.model.config),
            'formal': trainer.execution_context.config.value if trainer.execution_context is not None else None}
