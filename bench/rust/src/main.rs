//! Dependency-free cross-language micro-benchmark harness for `Df64`.
//!
//! The input generation and the timing methodology are shared between the
//! Rust, Julia and Python harnesses; the canonical specification lives in
//! `bench/README.md`.  The harness deliberately does not use `criterion`:
//! it prints one CSV row per `(implementation, operation, mode)`.
//!
//! Run with, for example:
//!
//! ```text
//! RUSTFLAGS="-C target-feature=+fma,+avx2" \
//!     cargo run --release --manifest-path bench/rust/Cargo.toml \
//!         -- --out bench/out/rust.csv
//! ```
//!
//! Copyright (C) 2023-2025 Markus Wallerberger and others
//! SPDX-License-Identifier: MIT

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use num_traits::Float;
use xprec::Df64;

// ---------------------------------------------------------------------------
// Shared input specification.  Must match `bench/julia/bench.jl` and
// `bench/python/bench.py` bit for bit; the FNV-1a checksum is compared by
// `bench/compare.py`.

const SPLITMIX_INC: u64 = 0x9E37_79B9_7F4A_7C15;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Exponent used for the `powi` benchmark in all implementations.
const POWI_EXP: i32 = 3;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INC);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv1a64(s: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Uniform value in `[1, 2)` built from the 52 low bits of `u`.
#[inline(always)]
fn unit12(u: u64) -> f64 {
    f64::from_bits(0x3FF0_0000_0000_0000 | (u >> 12))
}

/// Uniform value in `[-0.5, 0.5)`.
#[inline(always)]
fn signed(u: u64) -> f64 {
    unit12(u) - 1.5
}

/// Exact power of two `2^-k` for `k` in `[0, 21)`.
#[inline(always)]
fn pow2_neg(k: usize) -> f64 {
    1.0 / (1u64 << k) as f64
}

// ---------------------------------------------------------------------------
// Operations

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Noop,
    MulAdd,
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Powi,
    Powf,
    Sin,
    Cos,
    Tan,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Expm1,
    Log1p,
}

const OPS: &[(Op, &str)] = &[
    (Op::Noop, "noop"),
    (Op::MulAdd, "muladd"),
    (Op::Add, "add"),
    (Op::Sub, "sub"),
    (Op::Mul, "mul"),
    (Op::Div, "div"),
    (Op::Sqrt, "sqrt"),
    (Op::Exp, "exp"),
    (Op::Exp2, "exp2"),
    (Op::Log, "log"),
    (Op::Log2, "log2"),
    (Op::Log10, "log10"),
    (Op::Powi, "powi"),
    (Op::Powf, "powf"),
    (Op::Sin, "sin"),
    (Op::Cos, "cos"),
    (Op::Tan, "tan"),
    (Op::Atan, "atan"),
    (Op::Atan2, "atan2"),
    (Op::Sinh, "sinh"),
    (Op::Cosh, "cosh"),
    (Op::Tanh, "tanh"),
    (Op::Expm1, "expm1"),
    (Op::Log1p, "log1p"),
];

fn is_binary(op: Op) -> bool {
    matches!(
        op,
        Op::MulAdd | Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Atan2 | Op::Powf
    )
}

/// Primary input for `op` (see `bench/README.md`).
fn input_a(op: Op, u: u64) -> f64 {
    match op {
        Op::Exp | Op::Exp2 => signed(u) * 32.0,
        Op::Sin | Op::Cos | Op::Tan | Op::Atan | Op::Atan2 => signed(u) * 8.0,
        Op::Sinh | Op::Cosh | Op::Tanh => signed(u) * 4.0,
        Op::Expm1 | Op::Log1p => signed(u) * 0.0625,
        _ => unit12(u),
    }
}

/// Secondary input for `op` (see `bench/README.md`).  Generated for every
/// operation so that the timing loops stay uniform; unary operations ignore
/// it.
fn input_b(op: Op, i: usize, u: u64) -> f64 {
    match op {
        Op::Add | Op::Sub => unit12(u) * pow2_neg(i % 21),
        Op::Atan2 => signed(u) * 8.0,
        Op::Powf => signed(u) * 4.0,
        _ => unit12(u),
    }
}

/// Generate the inputs and the FNV-1a checksum over their bit patterns.
fn generate(op: Op, n: usize) -> (Vec<f64>, Vec<f64>, u64) {
    let mut state = fnv1a64(op_name(op));
    let mut hash = FNV_OFFSET;
    let update = |hash: &mut u64, x: f64| {
        let bits = x.to_bits();
        for k in 0..8 {
            *hash ^= (bits >> (8 * k)) & 0xFF;
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    let mut a = Vec::with_capacity(n);
    for _ in 0..n {
        let u = splitmix64(&mut state);
        let x = input_a(op, u);
        update(&mut hash, x);
        a.push(x);
    }

    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let u = splitmix64(&mut state);
        let x = input_b(op, i, u);
        update(&mut hash, x);
        b.push(x);
    }

    (a, b, hash)
}

fn op_name(op: Op) -> &'static str {
    OPS.iter().find(|(o, _)| *o == op).map(|(_, n)| *n).unwrap()
}

// ---------------------------------------------------------------------------
// Evaluation

#[inline(always)]
fn apply_f64(op: Op, a: f64, b: f64) -> f64 {
    match op {
        Op::Noop => a,
        Op::MulAdd => a.mul_add(b, 0.5),
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        Op::Sqrt => a.sqrt(),
        Op::Exp => a.exp(),
        Op::Exp2 => a.exp2(),
        Op::Log => a.ln(),
        Op::Log2 => a.log2(),
        Op::Log10 => a.log10(),
        Op::Powi => a.powi(POWI_EXP),
        Op::Powf => a.powf(b),
        Op::Sin => a.sin(),
        Op::Cos => a.cos(),
        Op::Tan => a.tan(),
        Op::Atan => a.atan(),
        Op::Atan2 => a.atan2(b),
        Op::Sinh => a.sinh(),
        Op::Cosh => a.cosh(),
        Op::Tanh => a.tanh(),
        Op::Expm1 => a.exp_m1(),
        Op::Log1p => a.ln_1p(),
    }
}

#[inline(always)]
fn apply_q(op: Op, a: Df64, b: Df64) -> Df64 {
    match op {
        Op::Noop => a,
        Op::MulAdd => Float::mul_add(a, b, Df64::from(0.5)),
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        Op::Sqrt => Float::sqrt(a),
        Op::Exp => Float::exp(a),
        Op::Exp2 => Float::exp2(a),
        Op::Log => Float::ln(a),
        Op::Log2 => Float::log2(a),
        Op::Log10 => Float::log10(a),
        Op::Powi => Float::powi(a, POWI_EXP),
        Op::Powf => Float::powf(a, b),
        Op::Sin => Float::sin(a),
        Op::Cos => Float::cos(a),
        Op::Tan => Float::tan(a),
        Op::Atan => Float::atan(a),
        Op::Atan2 => Float::atan2(a, b),
        Op::Sinh => Float::sinh(a),
        Op::Cosh => Float::cosh(a),
        Op::Tanh => Float::tanh(a),
        Op::Expm1 => Float::exp_m1(a),
        Op::Log1p => Float::ln_1p(a),
    }
}

/// A benchmarked value type: `f64` or `Df64`.
trait Value: Copy {
    const ONE: Self;

    /// Folds every limb into one `f64`, which the timing loops accumulate.
    ///
    /// This must depend on the whole result.  Accumulating only `hi` lets the
    /// compiler delete every instruction that feeds only `lo` (for example
    /// the error term of the final `fast_two_sum` of a multiplication), and
    /// the operation is then timed without part of its work.
    fn reduce(self) -> f64;
}

impl Value for f64 {
    const ONE: f64 = 1.0;

    #[inline(always)]
    fn reduce(self) -> f64 {
        self
    }
}

impl Value for Df64 {
    const ONE: Df64 = Df64::ONE;

    #[inline(always)]
    fn reduce(self) -> f64 {
        self.hi() + self.lo()
    }
}

// ---------------------------------------------------------------------------
// Timing

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    samples[samples.len() / 2]
}

/// Batched (throughput) form: independent operations with a four-way unrolled
/// accumulator, so that the accumulation itself does not dominate cheap
/// operations.  Iterating with `chunks_exact` keeps the indexing unchecked so
/// that the backend can vectorise the loop, matching what a Julia or NumPy
/// broadcast does.
fn time_throughput<T: Value>(a: &[T], b: &[T], reps: usize, f: impl Fn(T, T) -> T) -> f64 {
    let n = a.len();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut ca = a.chunks_exact(4);
        let mut cb = b.chunks_exact(4);
        let start = Instant::now();
        for (x, y) in ca.by_ref().zip(cb.by_ref()) {
            s0 += f(x[0], y[0]).reduce();
            s1 += f(x[1], y[1]).reduce();
            s2 += f(x[2], y[2]).reduce();
            s3 += f(x[3], y[3]).reduce();
        }
        for (x, y) in ca.remainder().iter().zip(cb.remainder().iter()) {
            s0 += f(*x, *y).reduce();
        }
        let dt = start.elapsed().as_nanos() as f64 / n as f64;
        black_box((s0, s1, s2, s3));
        samples.push(dt);
    }
    median(&mut samples)
}

/// Dependent (latency) form.  Informational only: several transcendental
/// operations degenerate (fixed point or NaN) when chained, so this is not
/// used for the regression threshold.
fn time_latency<T: Value>(
    a: &[T],
    b: &[T],
    reps: usize,
    binary: bool,
    f: impl Fn(T, T) -> T,
) -> f64 {
    let n = a.len();
    let mut samples = Vec::with_capacity(reps);
    let mut degenerate = false;
    for _ in 0..reps {
        let mut acc = T::ONE;
        let start = Instant::now();
        if binary {
            for i in 0..n {
                acc = f(black_box(a[i]), acc);
            }
        } else {
            for i in 0..n {
                acc = f(black_box(acc), black_box(b[i]));
            }
        }
        let dt = start.elapsed().as_nanos() as f64 / n as f64;
        degenerate |= !acc.reduce().is_finite();
        black_box(acc);
        samples.push(dt);
    }
    // A chain that drives the accumulator to infinity or NaN no longer
    // measures the operation, only the special-value branch that catches it.
    if degenerate {
        return f64::NAN;
    }
    median(&mut samples)
}

// ---------------------------------------------------------------------------
// Driver

struct Args {
    out: String,
    n: usize,
    reps: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        out: "bench/out/rust.csv".to_string(),
        n: 16384,
        reps: 15,
    };
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let take = |i: usize| -> String {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("missing value for {}", argv[i]))
                .clone()
        };
        match argv[i].as_str() {
            "--out" => args.out = take(i),
            "--n" => args.n = take(i).parse().unwrap(),
            "--reps" => args.reps = take(i).parse().unwrap(),
            other => panic!("unknown argument: {}", other),
        }
        i += 2;
    }
    args
}

fn main() {
    let args = parse_args();
    if let Some(parent) = Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).unwrap();
        }
    }

    let mut lines = vec![
        "# xprec-rs benchmark harness".to_string(),
        format!(
            "# impl=xprec n={} reps={} powi_exp={}",
            args.n, args.reps, POWI_EXP
        ),
        "impl,op,mode,ns_per_op,checksum".to_string(),
    ];

    println!("impl=xprec n={} reps={}", args.n, args.reps);
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>12} {:>18}",
        "op",
        "f64-thr",
        "f64-lat",
        "xprec-thr",
        "xprec-lat",
        "checksum"
    );

    for (op, name) in OPS {
        let (a64, b64, checksum) = generate(*op, args.n);

        let binary = is_binary(*op);
        let f_thr = time_throughput(&a64, &b64, args.reps, |x, y| apply_f64(*op, x, y));
        let f_lat = time_latency(&a64, &b64, args.reps, binary, |x, y| apply_f64(*op, x, y));

        let aq: Vec<Df64> = a64.iter().map(|x| Df64::from(*x)).collect();
        let bq: Vec<Df64> = b64.iter().map(|x| Df64::from(*x)).collect();
        let q_thr = time_throughput(&aq, &bq, args.reps, |x, y| apply_q(*op, x, y));
        let q_lat = time_latency(&aq, &bq, args.reps, binary, |x, y| apply_q(*op, x, y));

        println!(
            "{:<8} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:#018x}",
            name, f_thr, f_lat, q_thr, q_lat, checksum
        );

        let cs = format!("{checksum:#018x}");
        lines.push(format!("f64,{name},throughput,{f_thr:.6},{cs}"));
        lines.push(format!("f64,{name},latency,{f_lat:.6},{cs}"));
        lines.push(format!("xprec,{name}") + &format!(",throughput,{q_thr:.6},{cs}"));
        lines.push(format!("xprec,{name}") + &format!(",latency,{q_lat:.6},{cs}"));
    }

    fs::write(&args.out, lines.join("\n") + "\n").unwrap();
    println!("\nwrote {}", args.out);
}
