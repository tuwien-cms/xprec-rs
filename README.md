Fast emulated quadruple precision in Rust
=========================================

Emulates quadruple precision with a pair of doubles.  This roughly doubles
the mantissa bits (and thus squares the precision of double).  The range
is almost the same as double, with a larger area of denormalized numbers.
This is also called double-double arithmetic, compensated arithmetic, or Dekker
arithmetic.

The rough cost in floating point operations (fl) and relative error as
multiples of u² = 1.32e-32 (round-off error or half the machine epsilon) is
as follows:

  | (op)       | f64 f64 | error | Df64 f64 | error | Df64 Df64 | error |
  |------------|--------:|------:|--------:|------:|--------:|------:|
  | add_fast   |    3 fl |   0u² |    7 fl |   2u² |   17 fl |   3u² |
  | + -        |    6 fl |   0u² |   10 fl |   2u² |   20 fl |   3u² |
  | *          |    2 fl |   0u² |    6 fl |   2u² |    9 fl |   4u² |
  | /          |   3* fl |   1u² |   7* fl |   3u² |  28* fl |   6u² |
  | reciprocal |   3* fl |   1u² |         |       |  19* fl | 2.3u² |
  | sqrt       |   4* fl |   2u² |         |       |   8* fl |   4u² |

The error bounds are mostly tight analytical bounds (except for divisions).[^1]
An asterisk indicates the need for one or two double divisions, which are about
an order of magnitude more expensive than regular flops on a modern CPU.

The table can be distilled into two rules of thumb: double-double arithmetic
roughly doubles the number of significant digits at the cost of a roughly
15x slowdown compared to double arithmetic.

[^1]: M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018) and
      J.-M. Muller and L. Rideau, ACM Trans. Math. Softw. 48, 1, 9 (2022).
      The flop count has been reduced by 3 for divisons/reciprocals.
      In the case of double-double division, the bound is 10u² but largest
      observed error is 6u². In double by double division, we expect u². We
      report the largest observed error.


Benchmarks
----------
`bench/` compares `Df64` against [MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl)
(`Float64x2`) and the numpy `xprec` extension (`ddouble`) on batched elementwise
operations, and documents the methodology and its limitations.  The same
comparison runs on pull requests (see `.github/workflows/bench.yml`).


License and Copying
-------------------
Copyright (C) 2023-2025 Markus Wallerberger and others.

Released under the MIT license (see LICENSE for details).
