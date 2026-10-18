"""Test-only execution of the pinned author's actual entity-scoring AST block.

The seqeval import and token section are intentionally NOT executed. This is
not a claim that the complete upstream CLI or Tianchi server has been run.
"""

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_PATH = ROOT / "docs" / "前期核验" / "eval_task1_upstream.py"
AUTHOR_SHA256 = "9af0a1bdbdc813ff8f60b149c41e32d0d0b96f0cecb32b2bf0e37d568563d919"
AUTHOR_URL = "https://github.com/lemuria-wchen/imcs21-cblue/blob/8d512c6635029bf068712d62cf2428e65c5e4a58/task1/eval_task1.py"


def author_entity_reference(gold_data, pred_data):
    raw = AUTHOR_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUTHOR_SHA256:
        raise ValueError("saved author source hash differs from the pinned verified version")
    tree = ast.parse(raw.decode("utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    evaluate = functions["ner_eval"]
    # Keep every original statement before token_p=precision_score(...).
    cutoff = next(i for i, n in enumerate(evaluate.body)
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "token_p"
                                                      for t in n.targets))
    evaluate.body = evaluate.body[:cutoff] + ast.parse(
        "return {'overall_f1': overall_f1, 'class_info': class_info, "
        "'gold_total': origin, 'predicted_total': found, 'tp': right}").body
    module = ast.fix_missing_locations(ast.Module(
        body=[functions["get_entity_bio"], functions["compute"], evaluate], type_ignores=[]))
    namespace = {"Counter": Counter}
    exec(compile(module, str(AUTHOR_PATH) + ":entity_block_only", "exec"), namespace)
    output = io.StringIO()
    with redirect_stdout(output):
        result = namespace["ner_eval"](gold_data, pred_data)
    return {**result, "author_entity_stdout": output.getvalue(),
            "source_sha256": AUTHOR_SHA256, "source_url": AUTHOR_URL,
            "execution_scope": "get_entity_bio + compute + ner_eval entity prefix via AST; seqeval import/token section and full CLI NOT run"}
