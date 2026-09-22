#!/usr/bin/env python3
"""Merge the Rust / Julia / Python benchmark CSVs into a comparison report.

The report is meant to be read by humans *and* by an AI from the CI log, so it
is fully self-describing: environment, input checksums, per-operation numbers,
ratios and verdict are all present in one contiguous block.

Exit status is 0 unless a ratio exceeds ``--threshold``.

Usage:

    python bench/compare.py --rust out/rust.csv --julia out/julia.csv \
        --python out/python.csv --threshold 100 --json out/report.json

Copyright (C) 2023-2025 Markus Wallerberger and others
SPDX-License-Identifier: MIT
"""

import argparse
import json
import os
import sys

# Canonical operation order for the report.
OP_ORDER = [
    "noop", "muladd", "add", "sub", "mul", "div", "sqrt", "cbrt",
    "exp", "exp2", "log", "log2", "log10", "powi", "powf",
    "sin", "cos", "tan", "atan", "atan2",
    "sinh", "cosh", "tanh", "expm1", "log1p",
]

IMPL_ORDER = ["f64", "xprec", "multifloats", "numpy-xprec"]
BASELINES = ["multifloats", "numpy-xprec"]
REFERENCE = "xprec"

# Which implementation labels each harness provides.  Used to distinguish "the
# harness did not run" (an error) from "the library does not implement this
# operation" (not available).
SOURCE_IMPLS = {
    "rust": ["f64", "xprec"],
    "julia": ["multifloats"],
    "python": ["numpy-xprec"],
}

# Harness sanity check: without hardware FMA, `mul_add` falls back to libm and
# every Df64 result is inflated by roughly 3x.
CANARY_NUM = ("f64", "muladd")
CANARY_DEN = ("f64", "mul")
CANARY_MAX = 3.0

# Harness diagnostics, not library operations: excluded from the threshold and
# from the "worst ratio" summary.
DIAGNOSTIC_OPS = {"noop", "muladd"}


def read_csv(path):
    """Return `{(impl, op, mode): (ns, checksum)}` or None when unavailable."""
    if not path or not os.path.exists(path):
        return None
    data = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if parts[0] == "impl" or len(parts) != 5:
                continue
            impl, op, mode, ns, cs = parts
            data[(impl, op, mode)] = (float(ns), cs)
    return data


def fmt_ns(value):
    return f"{value:.3f}" if value is not None else "n/a"


def fmt_ratio(value):
    if value is None:
        return "n/a"
    if value >= 100.0:
        return f"{value:.0f}x"
    return f"{value:.2f}x"


def render_table(rows, value_of, columns, labels, ratio_names=()):
    """Render a fixed-width table as a list of lines."""
    width = max(len(label) for label in labels) + 2
    lines = [
        f"{'op':<8}" + "".join(f"{c:>{width}}" for c in labels),
        f"{'-' * 8}" + "-" * (width * len(labels)),
    ]
    for row in rows:
        line = f"{row['op']:<8}"
        for column in columns:
            line += f"{value_of(row, column):>{width}}"
        for name in ratio_names:
            line += f"{fmt_ratio(row['ratios'].get(name)):>{width}}"
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust", default="bench/out/rust.csv")
    parser.add_argument("--julia", default="bench/out/julia.csv")
    parser.add_argument("--python", default="bench/out/python.csv")
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--github", action="store_true",
                        help="emit ::error::/::warning:: workflow commands")
    parser.add_argument("--env", action="append", default=[],
                        help="extra `key=value` line for the report header")
    args = parser.parse_args()

    sources = {
        "rust": read_csv(args.rust),
        "julia": read_csv(args.julia),
        "python": read_csv(args.python),
    }
    table = {}
    for source in sources.values():
        if source:
            table.update(source)

    # A harness that produced no CSV at all is an environment failure: its
    # column is reported as an error, which is different from an operation the
    # library simply does not implement.
    failed = {name for name, src in sources.items() if src is None}
    failed_impls = {impl for name in failed for impl in SOURCE_IMPLS[name]}
    warnings = []
    for name in sorted(failed):
        warnings.append(
            f"{name} harness produced no CSV; "
            f"{', '.join(SOURCE_IMPLS[name])} recorded as error"
        )

    def get(impl, op, mode="throughput"):
        entry = table.get((impl, op, mode))
        return entry[0] if entry else None

    # --- input checksum agreement ------------------------------------------
    checksum_mismatch = []
    for op in OP_ORDER:
        seen = {
            impl: table[(impl, op, "throughput")][1]
            for impl in IMPL_ORDER
            if (impl, op, "throughput") in table
        }
        if len(set(seen.values())) > 1:
            checksum_mismatch.append({"op": op, "checksums": seen})
            warnings.append(f"INPUT MISMATCH for {op}: {seen}")

    # --- canary -------------------------------------------------------------
    canary = None
    num = get(*CANARY_NUM)
    den = get(*CANARY_DEN)
    if num is not None and den:
        canary = num / den
        if canary > CANARY_MAX:
            warnings.append(
                f"FMA appears DISABLED: f64 mul_add / f64 mul = {canary:.2f}x "
                f"(threshold {CANARY_MAX:.1f}x); every Df64 number below is inflated"
            )

    # --- throughput (used for the threshold) --------------------------------
    ops = []
    violations = []
    for op in OP_ORDER:
        row = {"op": op, "ns": {}, "ratios": {}}
        for impl in IMPL_ORDER:
            value = get(impl, op)
            if value is not None:
                row["ns"][impl] = value
        for baseline in BASELINES:
            ref = row["ns"].get(REFERENCE)
            base = row["ns"].get(baseline)
            if ref is not None and base is not None and base > 0:
                ratio = ref / base
                row["ratios"][baseline] = ratio
                if ratio > args.threshold and op not in DIAGNOSTIC_OPS:
                    violations.append({
                        "op": op,
                        "baseline": baseline,
                        "ratio": ratio,
                        "xprec_ns": ref,
                        "baseline_ns": base,
                    })
        if row["ns"]:
            ops.append(row)

    # --- latency (informational) --------------------------------------------
    latency = []
    for op in OP_ORDER:
        row = {"op": op, "ns": {}}
        for impl in IMPL_ORDER:
            value = get(impl, op, "latency")
            if value is not None:
                row["ns"][impl] = value
        if row["ns"]:
            latency.append(row)

    def fmt_cell(row, impl):
        if row["ns"].get(impl) is not None:
            return fmt_ns(row["ns"][impl])
        return "error" if impl in failed_impls else "n/a"

    def columns(rows):
        return [
            impl for impl in IMPL_ORDER
            if impl in failed_impls or any(impl in row["ns"] for row in rows)
        ]

    header = [f"{k}={v}" for k, v in (e.split("=", 1) for e in args.env)]

    throughput_cols = columns(ops)
    lines = []
    lines.append("================= xprec-rs benchmark comparison =================")
    for line in header:
        lines.append(line)
    lines.append("inputs         identical across implementations "
                 f"({len(checksum_mismatch)} checksum mismatches)")
    lines.append(f"threshold      {args.threshold:g}x (vs {'/'.join(BASELINES)})")
    lines.append("")
    lines.append("throughput (independent, batched; used for the threshold)")
    lines += render_table(
        ops, fmt_cell, throughput_cols,
        throughput_cols + [f"{REFERENCE}/{b}" for b in BASELINES],
        ratio_names=BASELINES,
    )
    if latency:
        latency_cols = columns(latency)
        lines.append("")
        lines.append("latency (dependent chain, informational; not used for the threshold)")
        lines += render_table(
            latency, fmt_cell, latency_cols, latency_cols)

    lines.append("")
    lines.append("NOTES")
    lines.append("  * units are ns per element, median of the harness repetitions")
    lines.append("  * the latency rows chain the operation into itself, which for")
    lines.append("    several transcendental functions degenerates to a fixed point")
    lines.append("  * n/a means the operation is not implemented by that library,")
    lines.append("    error means the harness for that column did not run")
    measured = {row["op"] for row in ops}
    gaps_by_impl = {}
    for impl in BASELINES + [REFERENCE]:
        if impl in failed_impls:
            continue
        # `noop` and `muladd` are harness diagnostics, not library operations,
        # so their absence must not be reported as an unimplemented feature.
        gaps = [op for op in OP_ORDER
                if op in measured and op not in DIAGNOSTIC_OPS
                and (impl, op, "throughput") not in table]
        if gaps:
            gaps_by_impl[impl] = gaps
    if gaps_by_impl:
        lines.append("  * per-implementation gaps:")
        for impl, gaps in gaps_by_impl.items():
            detail = " (unimplemented: `Float::cbrt` is `todo!()`)" if (
                impl == REFERENCE and gaps == ["cbrt"]) else ""
            lines.append(f"      {impl}: {', '.join(gaps)}{detail}")
    for op in sorted(DIAGNOSTIC_OPS):
        providers = [impl for impl in IMPL_ORDER
                     if (impl, op, "throughput") in table]
        if providers:
            lines.append(f"  * diagnostic row {op}: {', '.join(providers)}")
    if canary is not None:
        lines.append(f"  * canary f64 mul_add / f64 mul = {canary:.2f}x "
                     f"(>{CANARY_MAX:.1f}x means FMA is disabled)")
    for warning in warnings:
        lines.append(f"  * WARNING: {warning}")
    lines.append("")
    lines.append("VERDICT")
    if violations:
        for item in violations:
            lines.append(
                f"  FAIL {item['op']} is {item['ratio']:.1f}x slower than "
                f"{item['baseline']} ({item['xprec_ns']:.3f} vs "
                f"{item['baseline_ns']:.3f} ns/op)"
            )
        lines.append(f"  {len(violations)} (op, baseline) pair(s) exceed {args.threshold:g}x")
    else:
        worst = max(
            (r for row in ops if row["op"] not in DIAGNOSTIC_OPS
             for r in row["ratios"].values()),
            default=None,
        )
        lines.append(f"  OK: no (op, baseline) pair exceeds {args.threshold:g}x"
                     + (f"; worst ratio = {worst:.2f}x" if worst else ""))
    lines.append("=================================================================")

    print("\n".join(lines))

    if args.markdown:
        md = ["### throughput (used for the threshold)", "",
              "| op | " + " | ".join(throughput_cols) + " | "
              + " | ".join(f"{REFERENCE}/{b}" for b in BASELINES) + " |",
              "|" + "---|" * (1 + len(throughput_cols) + len(BASELINES))]
        for row in ops:
            cells = [row["op"]]
            cells += [fmt_cell(row, impl) for impl in throughput_cols]
            cells += [fmt_ratio(row["ratios"].get(b)) for b in BASELINES]
            md.append("| " + " | ".join(cells) + " |")
        if latency:
            latency_cols = columns(latency)
            md += ["", "### latency (informational)", "",
                   "| op | " + " | ".join(latency_cols) + " |",
                   "|" + "---|" * (1 + len(latency_cols))]
            for row in latency:
                cells = [row["op"]] + [fmt_cell(row, impl) for impl in latency_cols]
                md.append("| " + " | ".join(cells) + " |")
        md.append("")
        if violations:
            md.append(f"**{len(violations)} operation(s) exceed {args.threshold:g}x**")
            for item in violations:
                md.append(f"- `{item['op']}` is {item['ratio']:.1f}x slower than "
                          f"`{item['baseline']}`")
        else:
            md.append(f"No operation exceeds {args.threshold:g}x.")
        for warning in warnings:
            md.append(f"- WARNING: {warning}")
        if canary is not None:
            md.append(f"- canary `f64 mul_add / f64 mul` = {canary:.2f}x")
        path = args.markdown
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(md) + "\n")

    if args.json:
        report = {
            "env": dict(e.split("=", 1) for e in args.env),
            "threshold": args.threshold,
            "benchmarks": {row["op"]: {"ns": row["ns"], "ratios": row["ratios"]}
                           for row in ops},
            "latency": {row["op"]: row["ns"] for row in latency},
            "violations": violations,
            "warnings": warnings,
            "checksum_mismatches": checksum_mismatch,
            "missing_harness_output": {name: name in failed
                                       for name in sorted(SOURCE_IMPLS)},
            "failed_harness_impls": sorted(failed_impls),
        }
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    if args.github:
        for item in violations:
            print(f"::error title=benchmark::{item['op']} is {item['ratio']:.1f}x "
                  f"slower than {item['baseline']}")
        for warning in warnings:
            print(f"::warning title=benchmark::{warning}")

    if violations or checksum_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
