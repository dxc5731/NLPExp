"""Project-root launcher; explicit --preflight or --train remains mandatory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bert_crf.train import main

if __name__ == '__main__':
    raise SystemExit(main())
