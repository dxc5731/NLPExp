"""Reserved formal entry point. No implicit synthetic fallback."""
import argparse

from .training_config import TrainingConfig, require_synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = TrainingConfig.load(args.config)
    require_synthetic(config)
    raise RuntimeError('Formal entry disabled pending real Adapter integration; synthetic tests use python -m bert_crf.synthetic_smoke')


if __name__ == '__main__':
    main()
