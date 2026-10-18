"""Rebuild SMALL artificial handoff fixtures only; no training or inference."""
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    data = {"manual_scoring": {"dialogue": [
        {"sentence_id": "01", "sentence": "甲乙，丙丁。戊己，庚辛。壬癸。",
         "BIO_label": "B-Symptom I-Symptom O B-Drug I-Drug O B-Drug_Category I-Drug_Category O B-Medical_Examination I-Medical_Examination O B-Operation I-Operation O"},
        {"sentence_id": "09", "sentence": "A 😀，𠀀é", "BIO_label": "O O O O O O O"},
    ]}}
    write(ROOT / "data_dev.json", data)
    (ROOT / "fixture_source.txt").write_text("Synthetic interface fixture source marker v1. No model code or inference.\n", encoding="utf-8")
    sequences = {
        "rules": ("dictionary_rules", "人工规则接口", "O O O O O O O O O O O O O O O", "O O O O O O O"),
        "crf": ("crf", "人工CRF接口", "B-Symptom I-Symptom O O O O B-Drug_Category O O B-Drug I-Drug O B-Operation I-Operation O", "O O O O O O O"),
        "bert": ("bert", "人工BERT接口", "B-Symptom I-Symptom O B-Drug I-Drug O B-Drug_Category I-Drug_Category O B-Medical_Examination I-Medical_Examination O O O O", "B-Operation O O O O O O"),
    }
    runs = []
    for name, (kind, method, first, second) in sequences.items():
        directory = ROOT / name
        directory.mkdir(exist_ok=True)
        write(directory / "predictions_dev.json", {"manual_scoring": {"01": first, "09": second}})
        (directory / "fixture_creation.txt").write_text("完全人工构造的接口数据；未发生模型训练、推理或耗时测量。\n", encoding="utf-8")
        write(directory / "run_info.json", {
            "schema_version": "1.0", "artifact_kind": "synthetic_integration_test",
            "synthetic_notice": "人工三方法接入彩排；不是训练得到的规则、CRF 或 BERT 结果，运行专属 null 表示没有推理。",
            "run_id": "synthetic_" + name + "_v1", "method": {"name": method, "kind": kind},
            "method_version": "synthetic-interface-v1",
            "data": {"version": "manual-pipeline-fixture-v1", "split": "dev", "source_file": "../data_dev.json",
                     "source_sha256": digest(ROOT / "data_dev.json"), "archive_sha256": None},
            "code": {"revision": None, "dirty": None,
                     "snapshot": {"path": "../fixture_source.txt", "sha256": digest(ROOT / "fixture_source.txt")}},
            "config": {"version": "synthetic-v1", "inline": {"fixture": True, "model_inference": False}},
            "logs": [{"path": "fixture_creation.txt", "sha256": digest(directory / "fixture_creation.txt")}],
            "preprocessing": "无；人工原始字符接口样例。",
            "alignment": {"tokenizer": "未运行，人工接口", "character_mapping": "人工逐码点构造",
                          "special_tokens": "无", "long_text": "未执行分段或截断"},
            "prediction": {"file": "predictions_dev.json", "sha256": digest(directory / "predictions_dev.json")},
            "inference_command": None, "inference_working_directory": None, "device": None,
            "environment": {"python": platform.python_version(), "packages": {}, "context": "fixture_preparation_only"},
            "inference_elapsed_seconds": None, "timing_scope": None, "training": None, "random_seeds": None,
            "seed_applicability": "无训练／推理，人工确定性构造。",
        })
        runs.append({"run_id": "synthetic_" + name + "_v1", "method_name": method,
                     "predictions": name + "/predictions_dev.json", "run_info": name + "/run_info.json"})
    write(ROOT / "evaluation_runs.json", {"schema_version": "1.0", "purpose": "synthetic_integration_test",
        "data": {"path": "data_dev.json", "split": "dev"},
        "output_dir": "../../logs/pipeline_validation/batches", "runs": runs})
    print("Built three synthetic handoffs: 1 dialogue / 2 sentences / 22 code points each.")


if __name__ == "__main__":
    main()
