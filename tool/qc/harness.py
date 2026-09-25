"""CLI regression harness: run tool.qc.metrics over a fixture set and report
per-clip results as a Markdown-ish table + raw JSON, or diff two prior runs.

Usage:
    # existing output directory, one file per clip named "{short_name}_matte.gif"
    python3 -m tool.qc.harness --config v24 --fixture-set chroma \
        --output-dir results_dinosaur \
        --out /tmp/qc_v24.json

    python3 -m tool.qc.harness --compare /tmp/qc_v23.json /tmp/qc_v24.json

    # single explicit clip, bypassing fixture sets entirely
    python3 -m tool.qc.harness --config adhoc \
        --fixture 感謝=/path/src.mp4:/path/out.gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import fixtures as fx
from . import metrics as qc


def _resolve_output(clip_key, output_dir, name_template):
    short = fx.clip_short_name(clip_key) if "_" in clip_key else clip_key
    return Path(output_dir) / name_template.format(name=short)


def _run(config_name, fixtures_map, out_json):
    results = {}
    for clip, (src, out) in fixtures_map.items():
        print(f"[{config_name}] evaluating {clip} ({out}) ...", file=sys.stderr, flush=True)
        if not Path(out).exists():
            results[clip] = {"error": f"output file not found: {out}"}
            print(f"  SKIP: {results[clip]['error']}", file=sys.stderr)
            continue
        try:
            results[clip] = qc.evaluate(clip, str(src), str(out))
        except Exception as e:
            results[clip] = {"error": str(e)}
            print(f"  ERROR: {e}", file=sys.stderr)
    payload = {"config": config_name, "results": results}
    if out_json:
        Path(out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote {out_json}", file=sys.stderr)
    _print_table(payload)
    return payload


_METRIC_COLS = [
    ("F1_false_erase", "total"),
    ("F1i_false_erase_interior", "total"),
    ("F2_false_keep", "total"),
    ("F3_interior_holes", "total"),
    ("S1_mc_chatter", "mean_per_frame"),
    ("S2_color_flicker", "mean_per_frame"),
    ("S3_frozen_px", "total"),
    ("S4_area_jump", "worst_value"),
    ("E1_perimeter_ratio", "median"),
    ("E2_fringe_quality", "mean"),
]


def _print_table(payload):
    header = ["clip"] + [c for c, _ in _METRIC_COLS]
    rows = []
    for clip, r in payload["results"].items():
        if "error" in r:
            rows.append([clip, "ERROR: " + r["error"]] + [""] * (len(_METRIC_COLS) - 1))
            continue
        row = [clip]
        for metric, field in _METRIC_COLS:
            v = r[metric].get(field, "")
            row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        rows.append(row)
    widths = [max(len(str(h)), *(len(row[i]) for row in rows)) if rows else len(str(h))
              for i, h in enumerate(header)]

    def fmt(row):
        return " | ".join(str(c).ljust(w) for c, w in zip(row, widths))

    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _compare(path_a, path_b):
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    clips = sorted(set(a["results"]) & set(b["results"]))
    print(f"{'clip':<14} | {'metric':<24} | {a['config']:>12} | {b['config']:>12} | delta")
    for clip in clips:
        ra, rb = a["results"][clip], b["results"][clip]
        if "error" in ra or "error" in rb:
            print(f"{clip:<14} | ERROR ({ra.get('error') or rb.get('error')})")
            continue
        for metric, field in _METRIC_COLS:
            va, vb = ra[metric].get(field, 0), rb[metric].get(field, 0)
            delta = vb - va
            marker = ""
            if isinstance(delta, (int, float)) and abs(delta) > 1e-9:
                marker = "  (better)" if delta < 0 else "  (WORSE)"
            print(f"{clip:<14} | {metric + '.' + field:<24} | {va!s:>12} | {vb!s:>12} | {delta:+}{marker}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="label for this run, e.g. v23 / v24 / trimap")
    ap.add_argument("--fixture", action="append", default=[],
                     help="clip=source_mp4:output_file, repeatable; bypasses --fixture-set")
    ap.add_argument("--fixture-set", help="chroma / all "
                                           "(see tool/qc/fixtures.py)")
    ap.add_argument("--output-dir", help="directory holding '{name}_matte.gif' per clip")
    ap.add_argument("--output-name-template", default="{name}_matte.gif")
    ap.add_argument("--out", help="path to write JSON results")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()

    if args.compare:
        _compare(*args.compare)
        return

    fixtures_map = {}
    for f in args.fixture:
        clip, rest = f.split("=", 1)
        src, out = rest.split(":", 1)
        fixtures_map[clip] = (src, out)

    if args.fixture_set:
        sources = fx.get_sources(args.fixture_set)
        for clip, src in sources.items():
            out_dir = args.output_dir
            name_template = args.output_name_template
            if not out_dir:
                ap.error(f"--output-dir required to resolve output for {clip}")
            out = _resolve_output(clip, out_dir, name_template)
            fixtures_map[clip] = (src, out)

    if not fixtures_map:
        ap.error("no fixtures given (use --fixture and/or --fixture-set)")
    _run(args.config or "run", fixtures_map, args.out)


if __name__ == "__main__":
    main()
