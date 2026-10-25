"""Explicit formal preflight or train; neither action is the default."""
import argparse
from datetime import datetime
from pathlib import Path
import uuid

from .experiment import ROOT
from .formal_config import FormalConfig
from .preflight import run_preflight
from .trainer import Trainer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--train', action='store_true')
    parser.add_argument('--preflight-output')
    parser.add_argument('--resume', help='Trusted local epoch-boundary checkpoint; creates a new run')
    args = parser.parse_args(argv)
    if args.resume and not args.train:
        parser.error('--resume requires explicit --train')
    config = FormalConfig.load(args.config, root=ROOT)
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z') + '_' + uuid.uuid4().hex[:8]
    destination = Path(args.preflight_output) if args.preflight_output else ROOT / '.local/bert_crf/preflight' / stamp
    context = run_preflight(config, destination)
    print('PREFLIGHT PASS: ' + str(destination / 'preflight.json'))
    if args.preflight:
        return 0
    # The only CLI branch allowed to construct an updating Trainer.
    trainer = Trainer(context.model, config.training, context.providers['train'], context.providers['val'],
        config.path(config.value['outputs']['root']), execution_context=context,
        metadata={'experiment_name': config.value['experiment_name'], 'preflight_directory': str(destination.resolve())})
    trainer.run.event('formal_training_started', resume_from=args.resume)
    trainer.fit(resume_from=args.resume)
    print('TRAINING FINISHED: ' + str(trainer.run.path))
    return 0


if __name__ == '__main__':
    main()
