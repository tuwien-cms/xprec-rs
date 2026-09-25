//! Basic algorithms of compensated arithmetic.
//!
//! Most of the basic numerical algorithms are directly lifted from:
//!  - M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018)
//!  - Karp, High Precision Division and Square Root (1993)
//
// Copyright (C) 2023-2025 Markus Wallerberger and others
// SPDX-License-Identifier: MIT

use super::Df64;

// ---------------------------------------------------------------------------
// Helper functions

// smallest positive number
const fn tiny() -> f64
{
    const TINY: f64 = 4.9406564584124654e-324;
    debug_assert!(TINY != 0.0);
    return TINY;
}

// ---------------------------------------------------------------------------
// double (op) double -> quad

#[inline]
pub fn addfast_dd(a: f64, b: f64) -> Df64
{
    // M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018)
    // Algorithm 1: cost 3 flops
    let s = a + b;
    let z = s - a;
    let t = b - z;
    return Df64 {hi: s, lo: t};
}

#[inline]
pub fn subfast_dd(a: f64, b: f64) -> Df64
{
    // Algorithm 1 with b -> -b: cost 3 flops
    let s = a - b;
    let z = a - s;
    let t = z - b;
    return Df64 {hi: s, lo: t};
}

#[inline]
pub fn add_dd(a: f64, b: f64) -> Df64
{
    // Algorithm 2: cost 6 flops
    let s = a + b;
    let aprime = s - b;
    let bprime = s - aprime;
    let delta_a = a - aprime;
    let delta_b = b - bprime;
    let t = delta_a + delta_b;
    return Df64 {hi: s, lo: t};
}

#[inline]
pub fn sub_dd(a: f64, b: f64) -> Df64
{
    // Algorithm 2: cost 6 flops
    let s = a - b;
    let aprime = s + b;
    let bprime = aprime - s;
    let delta_a = a - aprime;
    let delta_b = bprime - b;
    let t = delta_a + delta_b;
    return Df64 {hi: s, lo: t};
}

#[inline]
pub fn mul_dd(a: f64, b: f64) -> Df64
{
    // Algorithm 3: cost 2 flops
    let pi = a * b;
    let rho = a.mul_add(b, -pi);
    return Df64 {hi: pi, lo: rho};
}

#[inline]
pub fn div_dd(a: f64, b: f64) -> Df64
{
    // Cost 3 flops (2 of which divisions), observed error 1 u^2
    // Since we are rounding faithfully, the hi part is exact
    let th = a / b;

    // Multiply hi part with b and compare exactly to a to see difference
    let rl = (-b).mul_add(th, a);
    let tl = rl / b;
    return Df64 {hi: th, lo: tl};
}

#[inline(always)]
pub fn reciprocal_d(x: f64) -> Df64
{
    return div_dd(1.0, x);
}

#[inline]
pub fn sqrt_d(a: f64) -> Df64
{
    // Karp, Table II, cost 5 flops, error 1 u^2
    let y0 = a.sqrt();

    // Adding a small number regularizes the case a == 0, where we would
    // otherwise divide zero by zero.
    let enumer = (-y0).mul_add(y0, a);
    let denom = tiny() + y0 + y0;
    let delta_y = enumer / denom;
    return Df64 {hi: y0, lo: delta_y};
}

// ---------------------------------------------------------------------------
// quad (op) double -> quad

#[inline]
pub fn addfast_qd(x: Df64, y: f64) -> Df64
{
    // Algorithm 4 modified: cost 7 flops, error 2 u^2
    let s = addfast_dd(x.hi, y);
    let v = x.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn subfast_qd(x: Df64, y: f64) -> Df64
{
    // Algorithm 4 modified: cost 7 flops, error 2 u^2
    let s = subfast_dd(x.hi, y);
    let v = x.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn add_qd(x: Df64, y: f64) -> Df64
{
    // Algorithm 4: cost 10 flops, error 2 u^2
    let s = add_dd(x.hi, y);
    let v = x.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn sub_qd(x: Df64, y: f64) -> Df64
{
    // Algorithm 4: cost 10 flops, error 2 u^2
    let s = sub_dd(x.hi, y);
    let v = x.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn mul_qd(x: Df64, y: f64) -> Df64
{
    // Algorithm 9: cost 6 flops, error 2 u^2
    let c = mul_dd(x.hi, y);
    let cl3 = x.lo.mul_add(y, c.lo);
    return addfast_dd(c.hi, cl3);
}

#[inline]
pub fn div_qd(x: Df64, y: f64) -> Df64
{
    // We could have used algorithm 15 here: cost 10 flops, error 3 u^2.
    // It turns out however by using fma, we can reduce this to 7 flops:
    //
    //    x / y = (x.hi + x.lo) / y = x.hi / y + x.lo / y .
    //
    // Defining the th = double(x.hi / y), we can rewrite this further as:
    //
    //    x / y = th + (x.hi - th * y) / y + x.lo / y ,
    //
    // where the second term can be computed to precision: f64 by fma, and
    // the together with the third term they are scaled by u, so are safe to
    // compute in precision: f64.
    let th = x.hi / y;
    let rl = (-y).mul_add(th, x.hi) + x.lo;
    let tl = rl / y;
    return addfast_dd(th, tl);
}

// ---------------------------------------------------------------------------
// quad (op) power of two -> quad

#[inline(always)]
pub fn add_pow2(a: Df64, p: f64) -> Df64
{
    // This can be added quickly because the mantissa part is zero.
    return addfast_qd(a, p);
}

#[inline(always)]
pub fn mul_pow2(a: Df64, p: f64) -> Df64
{
    return Df64 {hi: a.hi * p, lo: a.lo * p};
}

#[inline(always)]
pub fn div_pow2(a: Df64, p: f64) -> Df64
{
    return mul_pow2(a, 1.0 / p);
}

// ---------------------------------------------------------------------------
// double (op) quad -> quad

#[inline]
pub fn addfast_dq(x: f64, y: Df64) -> Df64
{
    // Algorithm 4 modified: cost 7 flops, error 2 u^2
    let s = addfast_dd(x, y.hi);
    let v = y.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn subfast_dq(x: f64, y: Df64) -> Df64
{
    // Algorithm 4 modified: cost 7 flops, error 2 u^2
    let s = subfast_dd(x, y.hi);
    let v = s.lo - y.lo;
    return addfast_dd(s.hi, v);
}

#[inline(always)]
pub fn add_dq(x: f64, y: Df64) -> Df64
{
    return add_qd(y, x);
}

#[inline(always)]
pub fn sub_dq(x: f64, y: Df64) -> Df64
{
    return add_qd(neg_q(y), x);
}

#[inline(always)]
pub fn mul_dq(x: f64, y: Df64) -> Df64
{
    return mul_qd(y, x);
}

#[inline(always)]
pub fn div_dq(x: f64, y: Df64) -> Df64
{
    return mul_qd(reciprocal_q(y), x);
}

// ---------------------------------------------------------------------------
// quad (op) quad -> quad

#[inline]
pub fn addfast_qq(x: Df64, y: Df64) -> Df64
{
    // Algorithm 6: cost 17 flops, error 3 u^2 + 13 u^3
    let s = addfast_dd(x.hi, y.hi);
    let t = add_dd(x.lo, y.lo);
    let c = s.lo + t.hi;
    let v = addfast_dd(s.hi, c);
    let w = t.lo + v.lo;
    return addfast_dd(v.hi, w);
}

#[inline]
pub fn subfast_qq(x: Df64, y: Df64) -> Df64
{
    // Algorithm 6: cost 17 flops, error 3 u^2 + 13 u^3
    let s = subfast_dd(x.hi, y.hi);
    let t = sub_dd(x.lo, y.lo);
    let c = s.lo + t.hi;
    let v = addfast_dd(s.hi, c);
    let w = t.lo + v.lo;
    return addfast_dd(v.hi, w);
}

#[inline]
pub fn add_qq(x: Df64, y: Df64) -> Df64
{
    // Algorithm 6: cost 20 flops, error 3 u^2 + 13 u^3
    let s = add_dd(x.hi, y.hi);
    let t = add_dd(x.lo, y.lo);
    let c = s.lo + t.hi;
    let v = addfast_dd(s.hi, c);
    let w = t.lo + v.lo;
    return addfast_dd(v.hi, w);
}

#[inline]
pub fn sub_qq(x: Df64, y: Df64) -> Df64
{
    // Algorithm 6: cost 20 flops, error 3 u^2 + 13 u^3
    let s = sub_dd(x.hi, y.hi);
    let t = sub_dd(x.lo, y.lo);
    let c = s.lo + t.hi;
    let v = addfast_dd(s.hi, c);
    let w = t.lo + v.lo;
    return addfast_dd(v.hi, w);
}

#[inline]
pub fn mul_qq(x: Df64, y: Df64) -> Df64
{
    // Algorithm 12: cost 9 flops, error 4 u^2 (corrected)
    let c = mul_dd(x.hi, y.hi);
    let tl0 = x.lo * y.lo;
    let tl1 = x.hi.mul_add(y.lo, tl0);
    let cl2 = x.lo.mul_add(y.hi, tl1);
    let cl3 = c.lo + cl2;
    return addfast_dd(c.hi, cl3);
}

#[inline]
pub fn div_qq(x: Df64, y: Df64) -> Df64
{
    return mul_qq(reciprocal_q(y), x);
}

#[inline(always)]
pub fn neg_q(x: Df64) -> Df64
{
    return Df64 {hi: -x.hi, lo: -x.lo};
}

#[inline]
pub fn reciprocal_q(y: Df64) -> Df64
{
    // Part of Algorithm 18: cost 19 flops, error 2.3 u^2
    let th = 1.0 / y.hi;
    let rh = (-y.hi).mul_add(th, 1.0);
    let rl = -y.lo * th;
    let e = addfast_dd(rh, rl);
    let delta = mul_qd(e, th);

    // This saves 3 flops w.r.t. algorithm 18, which uses standard addition.
    // We should be able to do this since Taylor expanding gives:
    //
    //  1/(xh + u*xl) = th * (1 + rh/th) * (1 + u * xl/xh + ...)
    //
    return addfast_dq(th, delta);
}

#[inline]
pub fn sqrt_q(a: Df64) -> Df64
{
    // Karp, Table II, cost 9 flops, error 2 u^2
    // The double result provides a approximation to sqrt(a). It performs
    // all the special-case handling, which is why we defer to it in these
    // cases.
    let y0 = a.hi.sqrt();

    // This is based on Newton-Ralphson for f(x) = a - 1/x^2:
    //
    //   x0 = approx(1/sqrt(A))
    //   x  = x + 0.5 * x * (1.0 - A * x * x)
    //
    let enumer = (-y0).mul_add(y0, a.hi) + a.lo;
    let denom = y0 + y0 + tiny();
    let delta_y = enumer / denom;

    // delta_y may alter the least significant digit of y0.
    return addfast_dd(y0, delta_y);
}

#[inline]
pub fn square_q(x: Df64) -> Df64
{
    // Simple squaring algorithm
    // Cost 7 flops
    let y = mul_dd(x.hi, x.hi);
    let y_lo = (x.lo + x.lo).mul_add(x.hi, y.lo);
    return addfast_dd(y.hi, y_lo);
}

/// Fused multiply-add for Df64: computes (x * y) + z
///
/// Note: There are two requirements for fma: (1) it must be accurate without
/// intermediate rounding and (2) it must be at least as fast as (x*y)+z.
/// For double-double, we cannot satisfy both, so we prioritize performance.
#[inline(always)]
pub fn mul_add_qq(x: Df64, y: Df64, z: Df64) -> Df64
{
    add_qq(mul_qq(x, y), z)
}

// ---------------------------------------------------------------------------
// UNIT TESTS

#[cfg(test)]
mod test
{
    use super::*;
    use crate::*;
    use super::super::test_utils::*;
    use crate::test_utils::PREC;
    use rug::Float;

    #[test]
    fn test_arith_dd()
    {
        let mut x = 10.0;
        while x > 5.0 {
            let mut y = x;
            while y > 1e-35 {
                // fast addition
                check_binary(addfast_dd, |x, y| x + y, x, y, 0.1);
                check_binary(addfast_dd, |x, y| x + y, -x, y, 0.1);
                check_binary(addfast_dd, |x, y| x + y, x, -y, 0.1);

                // fast subtraction
                check_binary(subfast_dd, |x, y| x - y, x, y, 0.1);
                check_binary(subfast_dd, |x, y| x - y, -x, y, 0.1);
                check_binary(subfast_dd, |x, y| x - y, x, -y, 0.1);

                // addition
                check_binary(add_dd, |x, y| x + y, x, y, 0.1);
                check_binary(add_dd, |x, y| x + y, y, x, 0.1);
                check_binary(add_dd, |x, y| x + y, x, -y, 0.1);
                check_binary(add_dd, |x, y| x + y, y, -x, 0.1);

                // subtraction
                check_binary(sub_dd, |x, y| x - y, x, y, 0.1);
                check_binary(sub_dd, |x, y| x - y, y, x, 0.1);
                check_binary(sub_dd, |x, y| x - y, x, -y, 0.1);
                check_binary(sub_dd, |x, y| x - y, y, -x, 0.1);

                // multiplication
                check_binary(mul_dd, |x, y| x * y, x, y, 0.1);
                check_binary(mul_dd, |x, y| x * y, x, -y, 0.1);

                // division
                check_binary(div_dd, |x, y| x / y, x, y, 1.0);
                check_binary(div_dd, |x, y| x / y, -x, y, 1.0);
                check_binary(div_dd, |x, y| x / y, y, x, 1.0);
                check_binary(div_dd, |x, y| x / y, -y, x, 1.0);

                y *= 0.9383;
            }
            x *= 0.9933;
        }
    }

    #[test]
    fn test_arith_qd()
    {
        let mut x = Df64::from(10.0);
        while x > Df64::from(5.0) {
            let mut y = x;
            while y > Df64::from(1e-35) {
                // fast addition
                check_binary(addfast_qd, |x, y| x + y, x, y.hi, 1.6);
                check_binary(addfast_qd, |x, y| x + y, x, -y.hi, 1.6);
                check_binary(addfast_qd, |x, y| x + y, neg_q(x), y.hi, 1.6);
                check_binary(addfast_qd, |x, y| x + y, neg_q(x), -y.hi, 1.6);
                check_binary(addfast_dq, |x, y| x + y, x.hi, y, 1.6);
                check_binary(addfast_dq, |x, y| x + y, x.hi, neg_q(y), 1.6);
                check_binary(addfast_dq, |x, y| x + y, -x.hi, y, 1.6);
                check_binary(addfast_dq, |x, y| x + y, -x.hi, neg_q(y), 1.6);

                // fast subtraction
                check_binary(subfast_qd, |x, y| x - y, x, y.hi, 1.6);
                check_binary(subfast_qd, |x, y| x - y, x, -y.hi, 1.6);
                check_binary(subfast_qd, |x, y| x - y, neg_q(x), y.hi, 1.6);
                check_binary(subfast_qd, |x, y| x - y, neg_q(x), -y.hi, 1.6);
                check_binary(subfast_dq, |x, y| x - y, x.hi, y, 1.6);
                check_binary(subfast_dq, |x, y| x - y, x.hi, neg_q(y), 1.6);
                check_binary(subfast_dq, |x, y| x - y, -x.hi, y, 1.6);
                check_binary(subfast_dq, |x, y| x - y, -x.hi, neg_q(y), 1.6);

                // addition
                check_binary(add_qd, |x, y| x + y, x, y.hi, 1.6);
                check_binary(add_qd, |x, y| x + y, y, x.hi, 1.6);
                check_binary(add_dq, |x, y| x + y, x.hi, y, 1.6);
                check_binary(add_dq, |x, y| x + y, y.hi, x, 1.6);
                check_binary(add_qd, |x, y| x + y, x, -y.hi, 1.6);
                check_binary(add_qd, |x, y| x + y, y, -x.hi, 1.6);
                check_binary(add_dq, |x, y| x + y, x.hi, neg_q(y), 1.6);
                check_binary(add_dq, |x, y| x + y, y.hi, neg_q(x), 1.6);

                // subtraction
                check_binary(sub_qd, |x, y| x - y, x, y.hi, 1.6);
                check_binary(sub_qd, |x, y| x - y, y, x.hi, 1.6);
                check_binary(sub_dq, |x, y| x - y, x.hi, y, 1.6);
                check_binary(sub_dq, |x, y| x - y, y.hi, x, 1.6);
                check_binary(sub_qd, |x, y| x - y, x, -y.hi, 1.6);
                check_binary(sub_qd, |x, y| x - y, y, -x.hi, 1.6);
                check_binary(sub_dq, |x, y| x - y, x.hi, neg_q(y), 1.6);
                check_binary(sub_dq, |x, y| x - y, y.hi, neg_q(x), 1.6);

                // multiplication
                check_binary(mul_qd, |x, y| x * y, x, y.hi, 2.0);
                check_binary(mul_qd, |x, y| x * y, x, -y.hi, 2.0);
                check_binary(mul_qd, |x, y| x * y, y, x.hi, 2.0);
                check_binary(mul_qd, |x, y| x * y, y, -x.hi, 2.0);

                // division
                check_binary(div_qd, |x, y| x / y, x, y.hi, 3.0);
                check_binary(div_qd, |x, y| x / y, x, -y.hi, 3.0);
                check_binary(div_qd, |x, y| x / y, y, x.hi, 3.0);
                check_binary(div_qd, |x, y| x / y, y, -x.hi, 3.0);
                check_binary(div_dq, |x, y| x / y, x.hi, y, 3.0);
                check_binary(div_dq, |x, y| x / y, x.hi, neg_q(y), 3.0);
                check_binary(div_dq, |x, y| x / y, y.hi, x, 3.0);
                check_binary(div_dq, |x, y| x / y, y.hi, neg_q(x), 3.0);

                y = mul_qd(y,0.9383);
            }
            x = mul_qd(x, 0.9933);
        }
    }

    #[test]
    fn test_arith_qq()
    {
        let mut x = Df64::from(10.0);
        while x > Df64::from(5.0) {
            let mut y = x;
            while y > Df64::from(1e-35) {
                // fast addition
                check_binary(addfast_qq, |x, y| x + y, x, y, 1.6);
                check_binary(addfast_qq, |x, y| x + y, x, neg_q(y), 1.6);
                check_binary(addfast_qq, |x, y| x + y, neg_q(x), y, 1.6);

                // fast subtraction
                check_binary(subfast_qq, |x, y| x - y, x, y, 1.6);
                check_binary(subfast_qq, |x, y| x - y, x, neg_q(y), 1.6);
                check_binary(subfast_qq, |x, y| x - y, neg_q(x), y, 1.6);

                // addition
                check_binary(add_qq, |x, y| x + y, x, y, 1.6);
                check_binary(add_qq, |x, y| x + y, y, x, 1.6);
                check_binary(add_qq, |x, y| x + y, x, neg_q(y), 1.6);
                check_binary(add_qq, |x, y| x + y, y, neg_q(x), 1.6);

                // subtraction
                check_binary(sub_qq, |x, y| x - y, x, y, 1.6);
                check_binary(sub_qq, |x, y| x - y, y, x, 1.6);
                check_binary(sub_qq, |x, y| x - y, x, neg_q(y), 1.6);
                check_binary(sub_qq, |x, y| x - y, y, neg_q(x), 1.6);

                // multiplication
                check_binary(mul_qq, |x, y| x * y, x, y, 2.0);
                check_binary(mul_qq, |x, y| x * y, x, neg_q(y), 2.0);

                // division
                check_binary(div_qq, |x, y| x / y, x, y, 3.0);
                check_binary(div_qq, |x, y| x / y, neg_q(x), y, 3.0);
                check_binary(div_qq, |x, y| x / y, y, x, 3.0);
                check_binary(div_qq, |x, y| x / y, neg_q(y), x, 3.0);

                y = mul_qd(y,0.9383);
            }
            x = mul_qd(x, 0.9933);
        }
    }

    #[test]
    fn test_arith_d()
    {
        check_unary(sqrt_d, |x| x.sqrt(), 0.0, 1.0);
        assert!(checks::is_nan(sqrt_d(-f64::MIN_POSITIVE)));

        let mut x = 1.0;
        while x > 1e-290 {
            check_unary(sqrt_d, |x| x.sqrt(), x, 2.0);
            check_unary(reciprocal_d, |x| 1.0 / x, x, 1.0);
            x *= 0.992;
        }

        x = 1.0;
        while x < 1e300 {
            check_unary(sqrt_d, |x| x.sqrt(), x, 2.0);
            if x < 1e290 {
                check_unary(reciprocal_d, |x| 1.0 / x, x, 1.0);
            }
            x /= 0.992;
        }
    }

    #[test]
    fn test_arith_q()
    {
        check_unary(sqrt_q, |x| x.sqrt(), Df64::ZERO, 1.0);
        assert!(checks::is_nan(sqrt_q(-Df64::MIN_POSITIVE)));

        let mut x = Df64::ONE;
        while x > Df64::from(1e-290) {
            check_unary(square_q, |x| x.clone() * x, sqrt_q(x), 2.0);
            check_unary(sqrt_q, |x| x.sqrt(), x, 2.0);
            check_unary(reciprocal_q, |x| 1.0 / x, x, 1.5);
            x = mul_qd(x, 0.992);
        }

        x = Df64::ONE;
        while x < Df64::from(1e300) {
            check_unary(square_q, |x| x.clone() * x, sqrt_q(x), 2.0);
            check_unary(sqrt_q, |x| x.sqrt(), x, 2.0);
            if x < Df64::from(1e290) {
                check_unary(reciprocal_q, |x| 1.0 / x, x, 1.5);
            }
            x = div_qd(x, 0.992);
        }
    }

    #[test]
    fn test_sum_stress(){
        let u = 0.5 * f64::EPSILON;

        let x = Df64 {hi: 1.0,  lo: u - u*u}    ;
        let y = Df64 {hi: 0.5 * (-1.0 + u), lo: u*u * (-0.5 + u)};
        let r: Df64 = x + y;
        let r_ex = Float::with_val(PREC, x) + Float::with_val(PREC, y);
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 3.0 * u * u) * r_ex.clone().abs();
            assert!(diff <= thr, "sum_stress: diff={} thr={}", diff, thr);
        }
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 2.5 * u * u) * r_ex.clone().abs();
            assert!(diff > thr, "sum_stress (neg): diff={} thr={}", diff, thr);
        }
    }

    // helper: integer ldexp
    fn ldexp_i(a: i64, e: i32) -> f64 {
        (a as f64) * (2f64).powi(e)
    }
    #[test]
    fn test_mul_stress(){


        let u = 0.5 * f64::EPSILON;
        let x = Df64 {hi: ldexp_i(2251799825991851, -51), lo: ldexp_i(9007199203085987, -106)};
        let y = Df64 {hi: ldexp_i(4503599627471459, -52), lo: ldexp_i(4503599627284651, -105)};

        let r = x * y;
        let r_ex = Float::with_val(PREC, x) * Float::with_val(PREC, y);
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 4.0 * u * u) * r_ex.clone().abs();
            assert!(diff <= thr, "mul_stress: diff={} thr={}", diff, thr);
        }
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 3.5 * u * u) * r_ex.clone().abs();
            assert!(diff > thr, "mul_stress (neg): diff={} thr={}", diff, thr);
        }
    }

    #[test]
    fn test_divdq_stress(){
        let u = 0.5 * f64::EPSILON;
        let x = Df64 {hi: 4588860379563012., lo: ldexp_i(-4474949195791253, -53)};
        let y = 4578284000230917.0;
        let r = x / y;
        let r_ex = Float::with_val(PREC, x) / Float::with_val(PREC, y);
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 3.0 * u * u) * r_ex.clone().abs();
            assert!(diff <= thr, "div_stress: diff={} thr={}", diff, thr);
        }
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 2.5 * u * u) * r_ex.clone().abs();
            assert!(diff > thr, "div_stress (neg): diff={} thr={}", diff, thr);
        }
    }

    #[test]
    fn test_divqq_stress(){
        let u = 0.5 * f64::EPSILON;
        let x = Df64 {hi: 4528288502329187.0 , lo: ldexp_i(1125391118633487, -51)};
        let y = Df64 {hi: 4522593432466394.0, lo: ldexp_i(-9006008290016505, -54)};
        let r = x / y;
        let r_ex = Float::with_val(PREC, x) / Float::with_val(PREC, y);
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 6.0 * u * u) * r_ex.clone().abs();
            assert!(diff <= thr, "div_stress: diff={} thr={}", diff, thr);
        }
        {
            let rr = Float::with_val(PREC, r);
            let diff = Float::with_val(PREC, &rr - &r_ex).abs();
            let thr  = Float::with_val(PREC, 0.9 * u * u) * r_ex.clone().abs();
            assert!(diff > thr, "div_stress (neg): diff={} thr={}", diff, thr);
        }
    }

}
