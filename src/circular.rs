use std::f64;

use super::Df64;
use super::arith::{
    addfast_dq, addfast_qd, addfast_qq, mul_dd, mul_pow2, reciprocal_q, sqrt_q, square_q,
    subfast_dq, subfast_qq,
};
use super::checks::{is_finite, is_nan, is_zero};
use super::consts;
use super::funcs::{abs, copysign};
use super::utils::reciprocal_factorial;

pub fn sin(x: Df64) -> Df64 {
    let (sector, z) = reduce_mod_pi2(x);
    match sector {
        0 => sin_kernel(z),
        1 => cos_kernel(z),  // sin(x) = cos(x - pi/2)
        2 => -sin_kernel(z), // sin(x) = -sin(x - pi)
        3 => -cos_kernel(z), // sin(x) = -cos(x + pi/2)
        _ => panic!("illegal sector"),
    }
}

pub fn cos(x: Df64) -> Df64 {
    let (sector, z) = reduce_mod_pi2(x);
    match sector {
        0 => cos_kernel(z),
        1 => -sin_kernel(z),
        2 => -cos_kernel(z),
        3 => sin_kernel(z),
        _ => panic!("illegal sector"),
    }
}

pub fn sincos(x: Df64) -> (Df64, Df64) {
    let (sector, z) = reduce_mod_pi2(x);
    let (s, c) = sincos_kernel(z);
    match sector {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        3 => (-c, s),
        _ => panic!("illegal sector"),
    }
}

pub fn tan(x: Df64) -> Df64 {
    let (s, c) = sincos(x);
    return s / c;
}

fn reduce_mod_pi2(x: Df64) -> (i32, Df64) {
    const INV_PI_HALF: f64 = 2.0 / f64::consts::PI;
    const PI_HALF: f64 = 1.5707963267948966;
    const PI_HALF_CORR: Df64 = Df64 {
        hi: 6.123233995736766e-17,
        lo: -1.4973849048591698e-33,
    };

    // Approximate reduction
    let n = (INV_PI_HALF * x.hi).round();
    let sector = (n as i32) & 0b11;

    // Do not perform reduction if no-op
    if n == 0.0 {
        return (0, x);
    }

    // The reduction is done in sextuple precision. This is slow but relatively
    // accurate.
    let z0 = subfast_qq(x, mul_dd(n, PI_HALF));
    let z = subfast_qq(z0, n * PI_HALF_CORR);
    return (sector, z);
}

fn sin_kernel(x: Df64) -> Df64 {
    // Taylor series of the sin around 0
    assert!(x.hi.abs() <= 0.7854);
    const NQUAD: i32 = 7;
    const N: i32 = 13;

    let xsq = -square_q(x);

    // r = x
    let mut r = x;
    let mut xpow = x;

    // r += x * (-x*x)**(i) / (2i+1)!
    for i in 1..NQUAD + 1 {
        xpow *= xsq;
        r = addfast_qq(r, reciprocal_factorial(2 * i + 1) * xpow);
    }

    // Here the terms are so small that they only affect the lo part, so
    // we can get away with double arithmetic.
    let xsq_d = xsq.hi;
    let mut xpow_d = xpow.hi;
    let mut r_d = 0.0;
    for i in NQUAD + 1..N + 1 {
        xpow_d *= xsq_d;
        r_d += reciprocal_factorial(2 * i + 1).hi * xpow_d;
    }

    // Add results together
    r = addfast_qd(r, r_d);
    return r;
}

fn cos_kernel(x: Df64) -> Df64 {
    // Taylor series of the cosine around 0
    assert!(x.hi.abs() <= 0.7854);
    const NQUAD: i32 = 8;
    const N: i32 = 13;

    let xsq = -square_q(x);

    // r = 1 - x*x / 2
    let mut r = addfast_dq(1.0, mul_pow2(xsq, 0.5));
    let mut xpow = xsq;

    // r += (-x*x)**(i+1) / (2i)!
    for i in 2..NQUAD + 1 {
        xpow *= xsq;
        r = addfast_qq(r, reciprocal_factorial(2 * i) * xpow);
    }

    // Here the terms are so small that they only affect the lo part, so
    // we can get away with double arithmetic.
    let xsq_d = xsq.hi;
    let mut xpow_d = xpow.hi;
    let mut r_d = 0.0;
    for i in NQUAD + 1..N + 1 {
        xpow_d *= xsq_d;
        r_d += reciprocal_factorial(2 * i).hi * xpow_d;
    }

    // Add results together
    r = addfast_qd(r, r_d);
    return r;
}

fn sincos_kernel(x: Df64) -> (Df64, Df64) {
    let s = sin_kernel(x);
    let c = sqrt_q(subfast_dq(1.0, square_q(s)));
    return (s, c);
}

pub fn asin(x: Df64) -> Df64 {
    // Compute a approximation to double precision
    let y0 = x.hi.asin();
    if !y0.is_finite() {
        return Df64::from(y0);
    }

    // This is where Taylor fails
    if abs(x) == Df64::ONE {
        return copysign(consts::PI_HALF, x);
    }

    // Perform Taylor expansion:
    //
    //    asin(x) = asin(x0) + (x - x0) / sqrt(1 - x0**2)
    //            = y0 + (x - sin(y0)) / cos(y0)
    //
    // XXX this has problems around 1
    let (x0, w) = sincos(Df64::from(y0));
    let y = y0 + subfast_qq(x, x0) / w;
    return y;
}

pub fn acos(x: Df64) -> Df64 {
    // Compute a approximation to double precision
    let y0 = x.hi.acos();
    if !y0.is_finite() {
        return Df64::from(y0);
    }

    // This is where Taylor fails
    if x == Df64::ONE {
        return Df64::ZERO;
    } else if x == Df64::from(-1.0) {
        return consts::PI;
    }

    // Perform Taylor expansion:
    //
    //    acos(x) = acos(x0) - (x - x0) / sqrt(1 - x0**2)
    //            = y0 - (x - cos(y0)) / sin(y0)
    //
    // XXX this has problems around 1
    let (w, x0) = sincos(Df64::from(y0));
    let y = y0 + subfast_qq(x0, x) / w;
    return y;
}

pub fn atan(x: Df64) -> Df64 {
    // For large values, use reflection formula
    if !(x.hi.abs() <= 1.0) {
        if is_nan(x) {
            return x;
        }
        let mut y = copysign(consts::PI_HALF, x);
        if is_finite(x) {
            y = subfast_qq(y, atan(reciprocal_q(x)));
        }
        return y;
    }

    // Again use Taylor expansion
    let y0 = x.hi.atan();
    let (s, c) = sincos(Df64::from(y0));
    let x0 = s / c;
    let y = addfast_dq(y0, subfast_qq(x, x0) * square_q(c));
    return y;
}

pub fn atan2(y: Df64, x: Df64) -> Df64 {
    // Special values
    if is_nan(x) || is_nan(y) {
        return Df64::NAN;
    } else if is_zero(y) {
        if x.hi >= 0.0 {
            return Df64::ZERO;
        } else {
            return consts::PI;
        }
    } else if is_zero(x) {
        return copysign(consts::PI_HALF, y);
    }

    let mut res = atan(y / x);
    if x.hi < 0.0 {
        res = addfast_qq(copysign(consts::PI, y), res);
    }
    return res;
}

#[cfg(test)]
mod test {
    use super::super::test_utils::*;
    use super::*;

    #[test]
    fn test_kernels() {
        // small values, start from PI/4
        let mut x = Df64::from(f64::consts::PI / 4.0);
        while x.hi > 1e-290 {
            check_unary(sin_kernel, |x| x.sin(), x, 1.1);
            check_unary(sin_kernel, |x| x.sin(), -x, 1.1);
            check_unary(cos_kernel, |x| x.cos(), x, 1.1);
            check_unary(cos_kernel, |x| x.cos(), -x, 1.1);

            check_unary(|x| sincos_kernel(x).0, |x| x.sin(), x, 1.1);
            check_unary(|x| sincos_kernel(x).0, |x| x.sin(), -x, 1.1);
            check_unary(|x| sincos_kernel(x).1, |x| x.cos(), x, 1.1);
            check_unary(|x| sincos_kernel(x).1, |x| x.cos(), -x, 1.1);
            x *= 0.947;
        }
    }

    #[test]
    fn test_circ() {
        // small values, start from PI/4
        let mut x = Df64::from(f64::consts::PI / 4.0);
        while x.hi > 1e-290 {
            check_unary(sin, |x| x.sin(), x, 1.1);
            check_unary(sin, |x| x.sin(), -x, 1.1);
            check_unary(cos, |x| x.cos(), x, 1.1);
            check_unary(cos, |x| x.cos(), -x, 1.1);
            check_unary(tan, |x| x.tan(), x, 2.0);
            check_unary(tan, |x| x.tan(), -x, 2.0);
            x *= 0.947;
        }

        // larger values
        x = Df64::from(f64::consts::PI / 4.0);
        while x.hi < 100.0 {
            let magn = x.hi.abs().max(1.0);
            check_unary(sin, |x| x.sin(), x, 1.5 * magn);
            check_unary(sin, |x| x.sin(), -x, 1.5 * magn);
            check_unary(cos, |x| x.cos(), x, 1.5 * magn);
            check_unary(cos, |x| x.cos(), -x, 1.5 * magn);
            check_unary(tan, |x| x.tan(), x, 2.5 * magn);
            check_unary(tan, |x| x.tan(), -x, 2.5 * magn);
            x /= 0.947;
        }
    }

    #[test]
    fn test_acirc() {
        let ulps = 1e-31 / Df64::EPSILON.hi;

        // asin
        check_unary(asin, |x| x.asin(), Df64::ZERO, ulps);
        check_unary(asin, |x| x.asin(), Df64::from(0.5), ulps);
        check_unary(asin, |x| x.asin(), Df64::from(-0.5), ulps);
        check_unary(asin, |x| x.asin(), Df64::from(1.0), ulps);
        check_unary(asin, |x| x.asin(), Df64::from(-1.0), ulps);
        // acos
        check_unary(acos, |x| x.acos(), Df64::ZERO, ulps);
        check_unary(acos, |x| x.acos(), Df64::from(0.5), ulps);
        check_unary(acos, |x| x.acos(), Df64::from(-0.5), ulps);
        check_unary(acos, |x| x.acos(), Df64::from(1.0), ulps);
        check_unary(acos, |x| x.acos(), Df64::from(-1.0), ulps);
        // atan
        check_unary(atan, |x| x.atan(), Df64::ZERO, ulps);
        check_unary(atan, |x| x.atan(), Df64::from(0.5), ulps);
        check_unary(atan, |x| x.atan(), Df64::from(-0.5), ulps);
        // atan2
        check_binary(atan2, |y, x| y.atan2(&x), Df64::ZERO, Df64::ZERO, ulps);
        check_binary(atan2, |y, x| y.atan2(&x), Df64::from(0.3), Df64::ZERO, ulps);
        check_binary(atan2, |y, x| y.atan2(&x), Df64::ZERO, Df64::from(1.0), ulps);
        check_binary(
            atan2,
            |y, x| y.atan2(&x),
            Df64::from(-0.5),
            Df64::ZERO,
            ulps,
        );
        check_binary(
            atan2,
            |y, x| y.atan2(&x),
            Df64::ZERO,
            Df64::from(-0.1),
            ulps,
        );

        check_binary(
            atan2,
            |y, x| y.atan2(&x),
            Df64::from(0.5),
            Df64::from(0.5),
            ulps,
        );
        check_binary(
            atan2,
            |y, x| y.atan2(&x),
            Df64::from(0.5),
            Df64::from(-0.5),
            ulps,
        );
        check_binary(
            atan2,
            |y, x| y.atan2(&x),
            Df64::from(-0.5),
            Df64::from(0.5),
            ulps,
        );
        check_binary(
            atan2,
            |y, x| y.atan2(&x),
            Df64::from(-0.5),
            Df64::from(-0.5),
            ulps,
        );

        // small values must be very accurate
        let mut x = Df64::from(1.0);
        while x > Df64::from(1e-300) {
            // asin
            check_unary(asin, |x| x.asin(), x, ulps);
            check_unary(asin, |x| x.asin(), -x, ulps);
            // acos
            check_unary(acos, |x| x.acos(), x, ulps);
            check_unary(acos, |x| x.acos(), -x, ulps);
            // atan
            check_unary(atan, |x| x.atan(), x, ulps);
            check_unary(atan, |x| x.atan(), -x, ulps);
            x *= 0.84;
        }

        let mut x = Df64::from(1.0);
        while x < Df64::from(1e290) {
            check_unary(atan, |x| x.atan(), x, ulps);
            check_unary(atan, |x| x.atan(), -x, ulps);
            x /= 0.84;
        }
    }
}
