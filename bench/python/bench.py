#!/usr/bin/env python3
"""Cross-language micro-benchmark harness (Python / numpy + xprec.ddouble).

The input generation and the timing methodology are shared with
``benches/core.rs`` and ``bench/julia/bench.jl``; the canonical specification
lives in ``bench/README.md``.

Only operations that the `xprec` numpy extension actually provides as ufuncs
are measured; `exp2`, `log2`, `log10` and `cbrt` have no `ddouble` ufunc and
are therefore reported as not available.

Usage:

    taskset -c 2 python bench/python/bench.py --out bench/out/python.csv

Copyright (C) 2023-2025 Markus Wallerberger and others
SPDX-License-Identifier: MIT
"""

import argparse
import os
import time

import numpy as np

import xprec

MASK = 0xFFFFFFFFFFFFFFFF
SPLITMIX_INC = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
POWI_EXP = 3

DD = xprec.ddouble
SINK = 0.0


# ---------------------------------------------------------------------------
# Shared input specification


def splitmix64(state):
    """Scalar splitmix64; used only for the per-operation seed."""
    state = (state + SPLITMIX_INC) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, z ^ (z >> 31)


def fnv1a64(text):
    h = FNV_OFFSET
    for byte in text.encode():
        h ^= byte
        h = (h * FNV_PRIME) & MASK
    return h


def _splitmix64_stream(seed, count):
    """Vectorised splitmix64 stream, identical to the scalar sequence."""
    k = np.arange(1, count + 1, dtype=np.uint64)
    state = np.uint64(seed) + k * np.uint64(SPLITMIX_INC)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def unit12(z):
    """Uniform value in [1, 2) built from the 52 low bits of `z`."""
    bits = np.uint64(0x3FF0000000000000) | (z >> np.uint64(12))
    return bits.view(np.float64)


def signed12(z):
    """Uniform value in [-0.5, 0.5)."""
    return unit12(z) - 1.5


def pow2_neg(k):
    """Exact power of two 2^-k for k in [0, 21)."""
    return 1.0 / (np.uint64(1) << k.astype(np.uint64)).astype(np.float64)


def input_a(op, z):
    if op in ("exp", "exp2"):
        return signed12(z) * 32.0
    if op in ("sin", "cos", "tan", "atan", "atan2"):
        return signed12(z) * 8.0
    if op in ("sinh", "cosh", "tanh"):
        return signed12(z) * 4.0
    if op in ("expm1", "log1p"):
        return signed12(z) * 0.0625
    return unit12(z)


def input_b(op, z, i):
    if op in ("add", "sub"):
        return unit12(z) * pow2_neg(i % 21)
    if op == "atan2":
        return signed12(z) * 8.0
    if op == "powf":
        return signed12(z) * 4.0
    return unit12(z)


def generate(op, n):
    seed = fnv1a64(op)
    z = _splitmix64_stream(seed, 2 * n)
    za, zb = z[:n], z[n:]
    idx = np.arange(n)
    a = input_a(op, za)
    b = input_b(op, zb, idx)
    return a, b, checksum(a, b)


def checksum(a, b):
    """FNV-1a over the little-endian bytes of all generated values."""
    h = FNV_OFFSET
    for byte in np.concatenate([a, b]).tobytes():
        h ^= byte
        h = (h * FNV_PRIME) & MASK
    return h


# ---------------------------------------------------------------------------
# Operations
#
# Each entry is `(n_arguments, callable)`.  `powi` uses the same fixed
# integer exponent as the other harnesses; note that numpy applies the
# general `power` ufunc here, so the Python number is not a dedicated
# integer-power kernel.

OPS = {
    "add": (2, lambda x, y: x + y),
    "sub": (2, lambda x, y: x - y),
    "mul": (2, lambda x, y: x * y),
    "div": (2, lambda x, y: x / y),
    "sqrt": (1, np.sqrt),
    "exp": (1, np.exp),
    "log": (1, np.log),
    "powi": (1, lambda x: np.power(x, POWI_EXP)),
    "powf": (2, np.power),
    "sin": (1, np.sin),
    "cos": (1, np.cos),
    "tan": (1, np.tan),
    "atan": (1, np.arctan),
    "atan2": (2, np.arctan2),
    "sinh": (1, np.sinh),
    "cosh": (1, np.cosh),
    "tanh": (1, np.tanh),
    "expm1": (1, np.expm1),
}


# ---------------------------------------------------------------------------
# Timing


def median_sample(samples):
    samples.sort()
    return samples[len(samples) // 2]


def time_batched(f, x, y, n, reps):
    """Batched form: one whole-array ufunc call, timed with `perf_counter_ns`."""
    global SINK
    f(x, y)  # warm up
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        out = f(x, y)
        t1 = time.perf_counter_ns()
        SINK += float(out.reshape(-1)[0])
        samples.append((t1 - t0) / n)
    return median_sample(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/out/python.csv")
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = ["impl,op,mode,ns_per_op,checksum"]

    print(f"impl=numpy-xprec n={args.n} reps={args.reps}")
    print(f"{'op':<8} {'py-thr':>12} {'checksum':>18}")

    for op, (nargs, f) in OPS.items():
        a64, b64, cs = generate(op, args.n)
        x = a64.astype(DD)
        y = b64.astype(DD)

        if nargs == 1:
            thr = time_batched(lambda p, q: f(p), x, y, args.n, args.reps)
        else:
            thr = time_batched(f, x, y, args.n, args.reps)

        print(f"{op:<8} {thr:12.4f} {cs:#018x}")
        rows.append(f"numpy-xprec,{op},throughput,{thr:.6f},{cs:#018x}")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("# xprec-rs benchmark harness (python/numpy+xprec)\n")
        handle.write(f"# impl=numpy-xprec n={args.n} reps={args.reps} powi_exp={POWI_EXP}\n")
        for row in rows:
            handle.write(row + "\n")
    print(f"\nwrote {args.out}")
    print(f"sink={SINK}")


if __name__ == "__main__":
    main()
