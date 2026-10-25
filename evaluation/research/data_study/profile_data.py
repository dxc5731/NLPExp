"""Aggregate fixed internal-train BIO statistics; never open dev/test members."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ner_metrics import extract_entities
from src.validate_predictions import ENTITY_TYPES, LOAD_FAILED, Report, build_data, parse_json, read_data

ARCHIVE_SHA = "84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67"
TRAIN_SHA = "07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24"
ENTITY_BINS = (("1", 1, 1), ("2", 2, 2), ("3–4", 3, 4), ("5–8", 5, 8), ("9+", 9, None))
SENTENCE_BINS = (("0–10", 0, 10), ("11–20", 11, 20), ("21–40", 21, 40),
                 ("41–80", 41, 80), ("81–160", 81, 160), ("161+", 161, None))
TYPE_NAMES = dict(zip(ENTITY_TYPES, ("症状", "药品", "药物类别", "医学检查", "医疗操作")))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return a / b if b else 0.0


def quantile(values, q):
    """Linear interpolation at (n-1)*q; empty data is null, not a fake zero."""
    if not 0 <= q <= 1:
        raise ValueError("quantile must be in [0,1]")
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def describe(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p90": quantile(values, .90), "p95": quantile(values, .95),
            "min": min(values) if values else None, "max": max(values) if values else None}


def bin_name(value, bins):
    for name, low, high in bins:
        if low <= value and (high is None or value <= high):
            return name
    raise ValueError("value outside declared bins")


def read_split(path):
    """Read a supplied manifest in place; do not extract/execute teammate code."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [i for i in archive.infolist() if not i.is_dir()
                       and Path(i.filename).name == "split_v1.json"]
            if len(members) != 1:
                raise ValueError("split archive must contain exactly one split_v1.json")
            raw = archive.read(members[0])
            member = members[0].filename
    else:
        raw, member = path.read_bytes(), None
    report = Report("train", True, 10)
    split = parse_json(raw, report, "data")
    if split is LOAD_FAILED or report.counts or not isinstance(split, dict):
        raise ValueError("split manifest must be valid UTF-8 JSON without duplicate keys")
    return split, {"path": str(path), "member": member, "container_sha256": sha256(path),
                   "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                   "version": split.get("version"), "seed": split.get("seed")}


def check_split(data, split, source_sha):
    if split.get("source_file") != "IMCS-V2_train.json" or split.get("source_sha256") != source_sha:
        raise ValueError("split manifest source file/hash does not match official train bytes")
    for name in ("train", "val"):
        ids = split.get(name)
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("split IDs must be nonempty lists of original string IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate dialogue ID in split manifest")
    train, val = set(split["train"]), set(split["val"])
    if train & val or train | val != set(data):
        raise ValueError("train/val must be disjoint and exactly cover official train dialogues")
    if split.get("n_dialogues") != len(data):
        raise ValueError("declared original dialogue count differs")
    # Val: verify membership and structural counts only, never inspect its BIO.
    counts = {}
    for name in ("train", "val"):
        rows = [r for did in split[name] for r in data[did]["dialogue"]]
        counts[name] = {"dialogues": len(split[name]), "sentences": len(rows),
                        "chars": sum(len(r["sentence"]) for r in rows)}
        expected = split.get("stats", {}).get(name, {})
        if not expected or any(expected.get(k) != value for k, value in counts[name].items()):
            raise ValueError("split manifest declared counts do not match source records")
    return {"status": "PASS", "disjoint": True, "complete": True, "counts": counts,
            "val_scope": "Membership/sentence/character counts only; no val BIO analysis."}


def analyze(data):
    """Aggregate validated raw dialogues; outputs contain no text or record IDs."""
    lengths, entity_lengths = [], []
    per_type_lengths = {kind: [] for kind in ENTITY_TYPES}
    per_type_sentences, count_hist, adjacency = Counter(), Counter(), Counter()
    sentence_rows = {name: Counter(sentences=0, entities=0, chars=0, entity_chars=0,
                                   multi_entity=0, multi_type=0, long_entity=0)
                     for name, _, _ in SENTENCE_BINS}
    total = Counter(sentences=0, chars=0, entities=0, entity_chars=0, no_entity=0,
                    multi_entity=0, multi_type=0, empty=0, long_entity=0,
                    adjacent_pairs=0, adjacent_sentences=0, direct_b_tags=0)
    for did, dialogue in data.items():
        for row in dialogue["dialogue"]:
            text, bio = row["sentence"], row["BIO_label"]
            tags = bio.split(" ") if bio else []
            if len(tags) != len(text):
                raise ValueError("tag/Unicode-code-point length mismatch")
            entities = sorted(extract_entities(tags, did, row["sentence_id"]), key=lambda e: e.start)
            n, n_chars = len(entities), len(text)
            kinds = {e.entity_type for e in entities}
            sizes = [e.end - e.start for e in entities]
            lengths.append(n_chars)
            entity_lengths.extend(sizes)
            per_type_sentences.update(kinds)
            count_hist[n] += 1
            for e in entities:
                per_type_lengths[e.entity_type].append(e.end - e.start)
            values = {"sentences": 1, "chars": n_chars, "entities": n,
                      "entity_chars": sum(sizes), "multi_entity": int(n >= 2),
                      "multi_type": int(len(kinds) >= 2), "long_entity": int(any(x >= 9 for x in sizes))}
            total.update(values)
            total["no_entity"] += int(n == 0)
            total["empty"] += int(n_chars == 0)
            total["direct_b_tags"] += sum(t.startswith("B-") for t in tags)
            sentence_rows[bin_name(n_chars, SENTENCE_BINS)].update(values)
            pairs = [(a.entity_type, b.entity_type) for a, b in zip(entities, entities[1:]) if a.end == b.start]
            adjacency.update(pairs)
            total["adjacent_pairs"] += len(pairs)
            total["adjacent_sentences"] += int(bool(pairs))
    type_rows = []
    length_rows = []
    for kind in ENTITY_TYPES:
        sizes = per_type_lengths[kind]
        type_rows.append({"entity_type": kind, "entities": len(sizes),
                          "entity_share": ratio(len(sizes), total["entities"]),
                          "sentences_containing_type": per_type_sentences[kind],
                          "sentence_share": ratio(per_type_sentences[kind], total["sentences"]),
                          "length": describe(sizes)})
    for kind, sizes in [("ALL", entity_lengths), *per_type_lengths.items()]:
        hist = Counter(bin_name(x, ENTITY_BINS) for x in sizes)
        length_rows.extend({"entity_type": kind, "length_bin": name, "count": hist[name],
                            "share_within_type": ratio(hist[name], len(sizes))}
                           for name, _, _ in ENTITY_BINS)
    complexity = []
    for name, low, high in SENTENCE_BINS:
        item = dict(sentence_rows[name])
        complexity.append({"length_bin": name, "low_inclusive": low, "high_inclusive": high,
                           **item, "sentence_share": ratio(item["sentences"], total["sentences"]),
                           "entities_per_100_chars": 100 * ratio(item["entities"], item["chars"]),
                           "entity_character_coverage": ratio(item["entity_chars"], item["chars"]),
                           "multi_entity_share": ratio(item["multi_entity"], item["sentences"]),
                           "multi_type_share": ratio(item["multi_type"], item["sentences"])})
    entity_counts = [r["entities"] for r in type_rows]
    checks = {
        "entities_equal_B_tags": total["entities"] == total["direct_b_tags"],
        "entity_type_sum": sum(entity_counts) == total["entities"],
        "entity_length_sum": sum(r["count"] for r in length_rows if r["entity_type"] == "ALL") == total["entities"],
        "sentence_bin_sum": sum(r["sentences"] for r in complexity) == total["sentences"],
        "entity_count_sentence_sum": sum(count_hist.values()) == total["sentences"],
        "entity_count_weighted_sum": sum(k * v for k, v in count_hist.items()) == total["entities"],
        "entity_chars_equal_non_O": total["entity_chars"] == sum(entity_lengths),
        "adjacency_pair_sum": sum(adjacency.values()) == total["adjacent_pairs"],
    }
    # Independent non-O count is calculated over annotations, not extracted spans.
    non_o = sum(sum(t != "O" for t in row["BIO_label"].split(" ") if t)
                for dialogue in data.values() for row in dialogue["dialogue"])
    checks["entity_chars_equal_non_O"] = total["entity_chars"] == non_o
    if not all(checks.values()):
        raise AssertionError("aggregate consistency check failed")
    return {"totals": {"dialogues": len(data), **dict(total),
                       "no_entity_share": ratio(total["no_entity"], total["sentences"]),
                       "multi_entity_share": ratio(total["multi_entity"], total["sentences"]),
                       "multi_type_share": ratio(total["multi_type"], total["sentences"]),
                       "o_character_share": 1 - ratio(total["entity_chars"], total["chars"]) if total["chars"] else 0.0,
                       "entity_character_coverage": ratio(total["entity_chars"], total["chars"]),
                       "largest_smallest_type_ratio": max(entity_counts) / min(entity_counts) if min(entity_counts) else None,
                       "adjacent_same_type_pairs": sum(n for (a, b), n in adjacency.items() if a == b)},
            "entity_types": type_rows, "entity_length_summary": describe(entity_lengths),
            "entity_length_bins": length_rows, "sentence_length_summary": describe(lengths),
            "sentence_length_bins": complexity,
            "entities_per_sentence": [{"entities": k, "sentences": v} for k, v in sorted(count_hist.items())],
            "adjacent_type_pairs": [{"left_type": a, "right_type": b, "pairs": n}
                                    for (a, b), n in sorted(adjacency.items())],
            "checks": checks}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_tables(stats, directory):
    tables = {"entity_types": [{**{k: v for k, v in row.items() if k != "length"},
                                **{"length_" + k: v for k, v in row["length"].items()}}
                               for row in stats["entity_types"]],
              "entity_lengths": stats["entity_length_bins"],
              "sentence_complexity": stats["sentence_length_bins"],
              "entities_per_sentence": stats["entities_per_sentence"],
              "adjacent_type_pairs": stats["adjacent_type_pairs"]}
    for name, rows in tables.items():
        # Explicit empty-table header; absence is never silently a missing file.
        fields = list(rows[0]) if rows else ["left_type", "right_type", "pairs"]
        path = directory / (name + ".csv")
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]
        if saved != expected:
            raise AssertionError("saved CSV differs from in-memory statistics")


def make_figures(stats, directory, scope):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib.font_manager import FontProperties, fontManager

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        fontManager.addfont(str(font))
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    colors = ["#3A79C0", "#82B6E8", "#B0D295", "#D0E4CC", "#E5F1E5"]
    ink, secondary, grid = "#263747", "#667783", "#E8EDF1"
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#AAB7C0", "axes.linewidth": .7,
        "axes.labelcolor": secondary, "axes.labelsize": 10.5,
        "axes.labelpad": 11, "axes.unicode_minus": False,
        "font.size": 10.5, "text.color": ink,
        "xtick.color": secondary, "ytick.color": secondary,
        "xtick.labelsize": 10, "ytick.labelsize": 10.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 9, "ytick.major.pad": 11,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    directory.mkdir()

    scope_label = "内部训练集" if scope == "internal_train" else scope
    totals = stats["totals"]

    def page(number, title, subtitle):
        fig = plt.figure(figsize=(11.2, 6.6))
        fig.text(.065, .945, f"{number:02d}  /  IMCS-V2-NER", fontsize=9,
                 color=secondary, va="top")
        fig.text(.065, .886, title, fontsize=20, weight="bold", va="top")
        fig.text(.065, .822, subtitle, fontsize=10.5, color=secondary, va="top")
        fig.add_artist(Line2D([.065, .945], [.785, .785], transform=fig.transFigure,
                              color=grid, linewidth=.8))
        fig.add_artist(Line2D([.065, .945], [.095, .095], transform=fig.transFigure,
                              color=grid, linewidth=.8))
        fig.text(.065, .055, f"{scope_label}  ·  {totals['dialogues']:,} 段对话 / {totals['sentences']:,} 句",
                 fontsize=8.5, color=secondary, va="center")
        fig.text(.945, .055, "训练数据统计 · 非模型效果", ha="right", va="center",
                 fontsize=8.5, color=secondary)
        return fig

    def style(ax, grid_axis="y"):
        ax.grid(axis=grid_axis, color=grid, linewidth=.8)
        ax.set_axisbelow(True)

    def save(fig, name):
        for extension in ("png", "svg", "pdf"):
            fig.savefig(directory / (name + "." + extension), dpi=300)
        plt.close(fig)

    labels = [f'{TYPE_NAMES[r["entity_type"]]}\n{r["entity_type"]}' for r in stats["entity_types"]]
    rows = stats["entity_types"]
    fig = page(1, "五类医疗实体的分布",
               f'共 {totals["entities"]:,} 个实体出现位置；按 BIO 跨度计数，不对实体词语去重')
    ax = fig.add_axes([.225, .205, .515, .515])
    ax.barh(range(len(rows)), [r["entities"] for r in rows], height=.57,
            color=colors, edgecolor="#93B2A3", linewidth=.55)
    ax.set_yticks(range(len(rows)), labels, color=ink, fontsize=10)
    ax.invert_yaxis()
    ax.set(xlabel="实体出现次数（个）", xlim=(0, max(r["entities"] for r in rows) * 1.065 or 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.spines["left"].set_visible(False)
    ax.text(1.16, 1.085, "实体数", transform=ax.transAxes, ha="right", color=secondary, fontsize=9.5)
    ax.text(1.37, 1.085, "占比", transform=ax.transAxes, ha="right", color=secondary, fontsize=9.5)
    for i, row in enumerate(rows):
        ax.text(1.16, i, f'{row["entities"]:,}', transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=11, weight="medium")
        ax.text(1.37, i, f'{row["entity_share"]:.2%}', transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=11, color=secondary)
    style(ax, "x")
    save(fig, "01_entity_distribution")

    fig = page(2, "不同类别的实体长度结构", "按各类别内部实体数归一化；每行合计为 100%")
    ax = fig.add_axes([.225, .275, .70, .445])
    left = [0.] * len(rows)
    for j, (color, (name, _, _)) in enumerate(zip(colors, ENTITY_BINS)):
        values = [next(x["share_within_type"] for x in stats["entity_length_bins"]
                       if x["entity_type"] == row["entity_type"] and x["length_bin"] == name) * 100 for row in rows]
        ax.barh(range(len(rows)), values, left=left, height=.62, color=color,
                edgecolor="white", linewidth=.8)
        for i, (value, start) in enumerate(zip(values, left)):
            if value >= 8:
                ax.text(start + value / 2, i, f"{value:.1f}%", ha="center", va="center",
                        color="white" if j == 0 else ink, fontsize=10)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(range(len(rows)), labels, color=ink, fontsize=10)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="各类别内部实体占比（%）")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.spines["left"].set_visible(False)
    style(ax, "x")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#93B2A3", linewidth=.55, label=name + " 字符")
                        for c, (name, _, _) in zip(colors, ENTITY_BINS)],
               loc="center", bbox_to_anchor=(.575, .145), ncol=5,
               frameon=False, fontsize=10, handlelength=1.8, columnspacing=1.8)
    save(fig, "02_entity_lengths")

    rows = stats["sentence_length_bins"]
    desc = stats["sentence_length_summary"]
    fig = page(3, "原始句长的分布",
               f'中位数 {desc["median"]:g} 字符  ·  P95 {desc["p95"]:g} 字符  ·  最大 {desc["max"]:g} 字符')
    ax = fig.add_axes([.115, .215, .81, .50])
    bars = ax.bar([r["length_bin"] for r in rows], [r["sentences"] for r in rows],
                  color=colors[0], width=.58, linewidth=0)
    for bar, row in zip(bars, rows):
        share = format(row["sentence_share"], ".2%" if row["sentence_share"] < .01 else ".1%")
        ax.annotate(f'{row["sentences"]:,}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 18), textcoords="offset points", ha="center", va="bottom", fontsize=11, weight="medium")
        ax.annotate(share, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=secondary)
    ax.set(xlabel="原始句长（Unicode 码点，包含空格与标点）", ylabel="句子数（千句）",
           ylim=(0, max(r["sentences"] for r in rows) * 1.24 or 1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    ax.yaxis.set_major_locator(MultipleLocator(10000))
    style(ax)
    save(fig, "03_sentence_lengths")

    fig = page(4, "句子中的实体结构与复杂度",
               f'无实体句 {totals["no_entity_share"]:.2%}  ·  多实体句 {totals["multi_entity_share"]:.2%}  ·  多类别句 {totals["multi_type_share"]:.2%}')
    axes = [fig.add_axes([.10, .235, .31, .43]), fig.add_axes([.535, .235, .40, .43])]
    hist = Counter({r["entities"]: r["sentences"] for r in stats["entities_per_sentence"]})
    values = [hist[0], hist[1], hist[2], hist[3], sum(n for k, n in hist.items() if k >= 4)]
    percentages = [100 * ratio(v, totals["sentences"]) for v in values]
    bars = axes[0].bar(["0", "1", "2", "3", "4+"], percentages, color=colors[0], width=.60)
    for bar, value in zip(bars, percentages):
        axes[0].annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                         xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    axes[0].set(xlabel="每句实体数（个）", ylabel="句子占比（%）", ylim=(0, 66))
    axes[0].set_title("A  每句实体数", loc="left", fontsize=12, weight="bold", pad=17)
    x = list(range(len(rows)))
    axes[1].plot(x, [100 * r["multi_entity_share"] for r in rows], "o-", color=colors[0],
                 linewidth=2.2, markersize=5.5, label="多实体（≥2 个）")
    axes[1].plot(x, [100 * r["multi_type_share"] for r in rows], "s--", color=colors[2],
                 linewidth=2.3, markersize=5.5, markeredgecolor="#688554", markeredgewidth=.7,
                 label="多类别（≥2 种）")
    axes[1].set_xticks(x, [f'{r["length_bin"]}\nn={r["sentences"]:,}' for r in rows], fontsize=8.7)
    axes[1].set(xlabel="原始句长区间（字符）", ylabel="区间内句子占比（%）", ylim=(0, 105))
    axes[1].set_title("B  不同句长下的实体结构", loc="left", fontsize=12, weight="bold", pad=17)
    axes[1].legend(frameon=False, loc="upper left", fontsize=9.2, handlelength=2.2)
    for ax in axes:
        style(ax)
    save(fig, "04_sentence_complexity")
    return {"matplotlib": matplotlib.__version__, "font": str(font) if font.exists() else "system default",
            "palette": colors, "style": "reference_bottom_right_blue_green",
            "formats": ["png", "svg", "pdf"], "png_dpi": 300}


def run(data_path, split_path, output_dir):
    started = time.perf_counter()
    report = Report("train", True, 15)
    data = read_data(data_path, "train", report)  # Only this archive member is opened.
    if data is LOAD_FAILED or report.counts or not isinstance(data, dict):
        raise ValueError("official train JSON could not be loaded/validated")
    source = report.inputs["data"]
    if source["sha256"] != TRAIN_SHA or source.get("archive_sha256", ARCHIVE_SHA) != ARCHIVE_SHA:
        raise ValueError("data hash differs from the fixed platform train version")
    if split_path:
        split, split_info = read_split(split_path)
        split_check = check_split(data, split, source["sha256"])
        selected = {did: data[did] for did in split["train"]}
        scope = "internal_train"
    else:
        split, split_info, split_check = None, None, {"status": "NOT_APPLICABLE", "reason": "Explicit official-train fallback."}
        selected, scope = data, "official_train_exploration"
    build_data(selected, report)
    validation = report.finish()
    if validation["status"] != "PASS":
        write_json(output_dir / "data_validation.json", validation)
        raise ValueError("source annotation audit failed; no statistics published")
    stats = analyze(selected)
    if split:
        expected = split["stats"]["train"]["entities"]
        if any(expected.get(r["entity_type"]) != r["entities"] for r in stats["entity_types"]):
            raise ValueError("independent extracted entity counts differ from supplied split metadata")
    definitions = {"character_unit": "Python Unicode code points including spaces/punctuation; no normalization",
                   "entity_unit": "BIO occurrence, strict extract_entities, [start,end); no string deduplication",
                   "entity_length_bins": ENTITY_BINS, "sentence_length_bins": SENTENCE_BINS,
                   "quantile": "linear interpolation at (n-1)*q", "multi_entity": ">=2 entities per sentence",
                   "multi_type": ">=2 distinct entity types per sentence", "long_entity": ">=9 code points",
                   "adjacent": "ordered consecutive spans with left.end == right.start",
                   "density": "entity occurrences per 100 sentence code points; separately report entity character coverage",
                   "ratios": "shares are fractions [0,1], empty-group shares=0; per-100 density and largest/smallest ratio are unbounded; missing smallest class ratio=null; empty distribution=null"}
    result = {"schema_version": "1.0", "purpose": "training_data_profile_not_model_evaluation", "status": "PASS",
              "scope": scope, "inputs": {"data": source, "split": split_info}, "split_check": split_check,
              "definitions": definitions, "statistics": stats, "annotation_validation": validation,
              "boundaries": {"dev_test_members_opened": False, "val_BIO_analyzed": False,
                             "medical_text_exported": False, "tokenizer_audit_repeated": False}}
    write_tables(stats, output_dir)
    result["plot_environment"] = make_figures(stats, output_dir / "figures", scope)
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(output_dir / "statistics.json", result)
    saved = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))
    if saved["statistics"] != stats:
        raise AssertionError("saved JSON statistics changed")
    hashes = {str(p.relative_to(output_dir)): sha256(p) for p in output_dir.rglob("*") if p.is_file()}
    write_json(output_dir / "run_manifest.json", {"status": "PASS", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "command": [sys.executable, *sys.argv], "python": platform.python_version(),
               "scripts": {str(Path(__file__).resolve()): sha256(__file__),
                           "src/validate_predictions.py": sha256(ROOT / "src/validate_predictions.py"),
                           "src/ner_metrics.py": sha256(ROOT / "src/ner_metrics.py")},
               "inputs": result["inputs"], "artifacts": hashes, "saved_JSON_CSV_self_check": "PASS"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--split-manifest", type=Path, help="Existing split JSON or interface ZIP containing it")
    choice.add_argument("--official-train", action="store_true", help="Explicit fallback; never labelled internal_train")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%dT%H%M%S_") + "profile_", dir=args.output_dir)).resolve()
    try:
        result = run(args.data, args.split_manifest, directory)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, ImportError, zipfile.BadZipFile) as exc:
        write_json(directory / "failure.json", {"status": "FAIL", "exception": type(exc).__name__, "message": str(exc),
                                               "notice": "No accepted statistics; input files were not repaired."})
        print(f"FAIL: {type(exc).__name__}; details: {directory}")
        return 1
    totals = result["statistics"]["totals"]
    print(f'PASS: {result["scope"]}; dialogues={totals["dialogues"]}, sentences={totals["sentences"]}, entities={totals["entities"]}')
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
