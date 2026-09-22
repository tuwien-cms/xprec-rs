Benchmark harness
=================

Cross-language micro-benchmarks that compare `xprec::Df64` against other
double-double implementations:

| implementation | language | type | effective mantissa |
|---|---|---|---|
| `xprec` | Rust | `Df64` = `Compensated<f64, f64>` | 106 bit |
| `multifloats` | Julia ([MultiFloats.jl]) | `Float64x2` | 106 bit |
| `numpy-xprec` | Python ([xprec] numpy extension) | `xprec.ddouble` | 106 bit |
| `f64` | Rust | `f64` | 53 bit (reference) |

[MultiFloats.jl]: https://github.com/dzhang314/MultiFloats.jl
[xprec]: https://github.com/tuwien-cms/xprec

The harnesses deliberately do **not** use `criterion`, `BenchmarkTools` or
`timeit`: each one is a small hand-rolled loop that prints a CSV with the
median over a fixed number of repetitions.  `@btime` would pull in a
dependency the harness does not otherwise need, and `timeit` disables the
garbage collector for the timed region (verified: `gc.isenabled()` is false
inside the region and restored afterwards), which would measure the Python
column under a different regime than the Julia one, whose collector stays
enabled; the Rust harness allocates on the stack and has none.  Using the
same shape of timer everywhere keeps the ratio the only thing that differs
between the columns.

There is no benchmark suite in either upstream project to compare against;
the performance table in the top-level `README.md` is the analytic flop count
from the CAMPARY papers, not a measurement.


Running
-------

```sh
# Rust
RUSTFLAGS="-C target-feature=+fma,+avx2" \
    taskset -c 2 cargo bench --bench core -- --out bench/out/rust.csv

# Julia (needs MultiFloats, see bench/julia/Project.toml)
julia --project=bench/julia -e 'using Pkg; Pkg.instantiate()'
taskset -c 2 julia --project=bench/julia bench/julia/bench.jl --out bench/out/julia.csv

# Python (needs numpy and the xprec numpy extension)
pip install -r bench/python/requirements.txt
taskset -c 2 python bench/python/bench.py --out bench/out/python.csv

# Merge
python bench/compare.py --rust bench/out/rust.csv \
    --julia bench/out/julia.csv --python bench/out/python.csv \
    --threshold 100 --json bench/out/report.json --markdown bench/out/report.md
```

`compare.py` exits non-zero when an operation is more than `--threshold`
times slower than a baseline, or when two harnesses disagree about the input
checksum.  In the report, `n/a` means "this library does not implement the
operation" and `error` means "the harness for this column did not run"; the
two are deliberately different, because only the second one invalidates the
comparison.  The same comparison runs on pull requests, see
`.github/workflows/bench.yml`.

> **`-C target-feature=+fma` is not optional.**  `Df64` arithmetic goes through
> `f64::mul_add`, which compiles to a call to libm's correctly-rounded `fma`
> unless the target enables the FMA instruction.  On an AMD EPYC 7713P this
> makes `Df64` roughly 3x slower for no algorithmic reason.  The report
> contains an `f64 mul_add / f64 mul` canary row that warns when this happens.


Methodology
-----------

**Batched elementwise loops, not per-call timings.**  A single call is
dominated by dispatch and call overhead, which differ wildly between the three
languages.  Every measurement applies the operation to an `N`-element array
(`N = 16384` by default) and reports ns per element.

| | batched form |
|---|---|
| Rust | `chunks_exact(4)` loop over two slices, four-way unrolled accumulator |
| Julia | closure per operation, scalar loop over `Vector{Float64x2}`, four-way unrolled |
| Python | one whole-array ufunc call (`np.exp(x)`, `x + y`, ...) |

Only the **throughput** (independent operations) form is used for the
threshold.  A **latency** (`acc <- op(a[i], acc)`) form is measured by the
Rust and Julia harnesses and printed as a second, informational table; it is
never compared, because chained transcendental operations degenerate to a
fixed point or to `NaN`.

Each harness reports the **median** over `--reps` (default 15) repetitions.
An empty-loop **`noop`** row gives the harness floor so that a reader can judge
how much of a cheap operation's cost is measurement overhead, and the
**`muladd`** row is the FMA canary.  Both are harness diagnostics rather than
library operations: they are excluded from the threshold and are never listed
as an unimplemented feature when a harness does not provide them.

**Identical inputs.**  All three harnesses generate the inputs from the same
integer recipe, so no file needs to be shipped and the values are bit-identical
by construction:

```
splitmix64(state):
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return state, z ^ (z >> 31)

seed(op)  = FNV-1a 64 of the operation name
            (offset 0xCBF29CE484222325, prime 0x100000001B3)
A         = first N splitmix64 draws from seed(op)
B         = next N draws
unit12(u) = reinterpret(f64, 0x3FF0000000000000 | (u >> 12))    # in [1, 2)
signed(u) = unit12(u) - 1.5                                     # in [-0.5, 0.5)
pow2_neg(k) = 1.0 / (1 << k)                                    # exact
```

Per-operation transforms (only `+ - * /` and exact powers of two, so the
results are identical under IEEE-754 round-to-nearest everywhere):

| operation | A | B |
|---|---|---|
| `add`, `sub` | `unit12` | `unit12 * pow2_neg(i % 21)` |
| `mul`, `div`, `sqrt`, `log`, `log2`, `log10`, `cbrt` | `unit12` | `unit12` |
| `exp`, `exp2` | `signed * 32` | `unit12` |
| `sin`, `cos`, `tan`, `atan` | `signed * 8` | `unit12` |
| `atan2` | `signed * 8` | `signed * 8` |
| `sinh`, `cosh`, `tanh` | `signed * 4` | `unit12` |
| `expm1`, `log1p` | `signed * 0.0625` | `unit12` |
| `powf` | `unit12` | `signed * 4` |
| `powi` | `unit12` | (unused; exponent is fixed at 3) |
| `noop`, `muladd` | `unit12` | `unit12` |

`B` is generated for every operation, including unary ones, so that the timing
loops stay uniform.

Each harness also prints an FNV-1a checksum over the bit patterns of `A`
followed by `B`.  `compare.py` fails the run when the checksums of two
implementations disagree for the same operation, which is the guard against
silently measuring different inputs.

**Matched input values are not matched distributions for free.**  Because the
same values reach every implementation, the comparison is fair, but the
distributions above are only one sample of the input space.  In particular
`add`/`sub` deliberately mix exponent gaps; there is no cancellation case, no
subnormal input and no `NaN`/`Inf` input (MultiFloats treats `Inf` as `NaN`).
Adding variants means extending all three harnesses and `compare.py` together.


Coverage
--------

Not every library implements every operation, so `compare.py` prints `n/a`
and lists the gaps:

* MultiFloats.jl has no `sin`/`cos`/`tan`/`atan`/`atan2`, no
  `sinh`/`cosh`/`tanh`, no `expm1`/`log1p` — these raise an error in the
  measured release (`_BASE_TRANSCENDENTAL_FUNCTIONS` in `src/MultiFloats.jl`).
* The numpy `xprec` extension has no `exp2`, `log2`, `log10`, `cbrt` or
  `log1p` ufunc.
* `xprec-rs` does not implement `Float::cbrt` (`todo!()`), so `cbrt` is only
  measured for MultiFloats.jl.
* `powi` uses a fixed integer exponent of 3 in all three harnesses.  Note that
  `xprec-rs` implements `powi` as `exp(n * log(x))` while MultiFloats.jl uses
  repeated squaring, so this row compares different algorithms; and numpy's
  `power` ufunc is the general (non-integer) path.


Known limitations
-----------------

* Everything runs single threaded (`JULIA_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, pinned to one core) so that the numbers do not measure
  how well each runtime parallelises.
* Cross-implementation ratios are used, never absolute timings, because the
  host CPU and its clock frequency are not controlled.  Run all three
  harnesses back to back on one machine, pinned to one core, for a meaningful
  comparison.  On the development machine the `schedutil` governor runs the
  EPYC 7713P at 1.5 GHz, which inflates every absolute number but cancels in
  the ratios.
* The three harnesses have different floors (the `noop` row): a scalar Rust
  loop is more expensive than a vectorised Julia loop, and a numpy ufunc call
  allocates a result array.  This matters for cheap operations (`add`, `mul`)
  and is invisible for the expensive ones that the 100x threshold targets.
* Rust is built with `-C target-feature=+fma,+avx2`; the `f64` baseline inside
  the Rust harness uses the same flags.
* `powi` and `powf` take early-exit branches for the IEEE-754 special values,
  so those two rows measure that branchy path rather than a straight
  `exp(n * log(x))`.  They also describe different algorithms than
  MultiFloats.jl (see Coverage above), so the ratios on those rows are not
  purely an implementation-quality comparison.
