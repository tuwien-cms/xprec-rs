use super::Df64;
use super::arith::{
    addfast_dq, addfast_qq, mul_pow2, reciprocal_q, sqrt_q, square_q, subfast_dq, subfast_qd,
    subfast_qq,
};
use super::checks::{is_finite, is_nan};
use super::exp::{exp_split, expm1, log, log1p};
use super::funcs::{abs, copysign, ldexp};
use super::roots::{hypot, inv_sqrt};

pub const COSH_MAX: f64 = 710.4758600739439;

pub fn cosh(x: Df64) -> Df64 {
    let xx = abs(x);
    if !(xx.hi <= COSH_MAX) {
        if is_nan(x) {
            return Df64::NAN;
        } else {
            return Df64::INFINITY;
        }
    }

    // simply use the definition:
    //
    //    cosh(x) = exp(x)/2 + exp(-x)/2 = exp(x)/2 + 1/4 * 2/exp(x)
    //
    let (m, y) = exp_split(xx);
    let exp_x_half = ldexp(addfast_dq(1.0, y), m - 1);

    // For large numbers, that's it.
    if !(xx.hi <= 36.5) {
        return exp_x_half;
    }

    let exp_mx_half = mul_pow2(reciprocal_q(exp_x_half), 0.25);
    return addfast_qq(exp_x_half, exp_mx_half);
}

pub fn sinh(x: Df64) -> Df64 {
    let xx = abs(x);
    if !(xx.hi <= COSH_MAX) {
        if is_nan(x) {
            return Df64::NAN;
        } else {
            return copysign(Df64::INFINITY, x);
        }
    }

    let (m, y) = exp_split(xx);

    // For small x, corresponding to exponent m = 0 of exp(x), naive
    // subtraction leads to cancellation. Let's instead rewrite. Let
    // exp(x) = 1 + y, then:
    //
    //    sinh(x) = y * (1 + 0.5 * y) / (1 + y)
    //
    if m == 0 {
        let dn = addfast_dq(1.0, mul_pow2(y, 0.5));
        let en = addfast_dq(1.0, y);
        return copysign(y * dn / en, x);
    }

    // Otherwise simply use the definition:
    //
    //    sinh(x) = exp(x)/2 - exp(-x)/2 = exp(x)/2 - 1/4 * 2/exp(x)
    //
    let exp_x_half = ldexp(addfast_dq(1.0, y), m - 1);

    // For large numbers, that's it.
    if !(xx.hi <= 36.5) {
        return copysign(exp_x_half, x);
    }

    let exp_mx_half = mul_pow2(reciprocal_q(exp_x_half), 0.25);
    return copysign(subfast_qq(exp_x_half, exp_mx_half), x);
}

pub fn tanh(x: Df64) -> Df64 {
    let xx = abs(x);
    if !(xx.hi <= 36.5) {
        if is_nan(x) {
            return Df64::NAN;
        } else {
            // Asymptotically, we have +- 1
            return copysign(Df64::ONE, x);
        }
    }

    // For small x, the naive tanh(x) leads to cancellation. Again, we rewrite
    // it. Let exp(x) = 1 + y, then:
    //
    //    tanh(x) = z / (2 + z)  with  z = y * (2 + y)
    //
    // This is stable along the entire range.
    let y = expm1(xx);
    let z = y * (2.0 + y);
    return copysign(z / (2.0 + z), x);
}

pub fn asinh(x: Df64) -> Df64 {
    // Special values: +Inf, -Inf are all preserved
    if !is_finite(x) {
        return x;
    }

    // For small values, use Taylor expansion around the double result,
    // because the bottom expression is log(1 + 2x/3 + ...), subject to
    // cancellation.
    if x.hi.abs() < 1.0 {
        let y0 = x.hi.asinh();
        let x0 = sinh(Df64::from(y0));

        let delta_y = (x - x0) * inv_sqrt(1.0 + square_q(x0));
        return addfast_dq(y0, delta_y);
    }

    // Use the definition:
    //
    //     asinh(x) = log(sqrt(1 + x^2) + x)
    //
    let xx = abs(x);
    let arg = addfast_qq(hypot(Df64::ONE, xx), xx);
    return copysign(log(arg), x);
}

pub fn acosh(x: Df64) -> Df64 {
    // Special values: +Inf, -Inf are all preserved
    if !is_finite(x) {
        return x;
    }

    // Use the definition:
    //
    //     acosh(x) = log(x + sqrt(x^2 - 1))
    //
    // but be careful of overflows.
    let arg = if x.hi < 1.0 / f64::EPSILON {
        addfast_qq(x, sqrt_q(subfast_qd(square_q(x), 1.0)))
    } else {
        mul_pow2(x, 2.0)
    };
    return log(arg);
}

pub fn atanh(x: Df64) -> Df64 {
    if is_nan(x) {
        return x;
    }

    // Special value
    let xx = abs(x);
    if xx == Df64::ONE {
        return copysign(Df64::INFINITY, x);
    }

    // Use the definition, but be wary of cancellation around 0.
    //
    //   atanh(x) = 1/2 log((1 + x)/(1 - x)) = 1/2 log(1 + 2x/(1 - x))
    //
    let twox = mul_pow2(xx, 2.0);
    let one_minus_x = subfast_dq(1.0, xx);
    return copysign(mul_pow2(log1p(twox / one_minus_x), 0.5), x);
}

#[cfg(test)]
mod test {
    use super::super::checks::is_infinite;
    use super::super::test_utils::*;
    use super::*;

    #[test]
    fn test_cosh() {
        // special values
        assert!(is_infinite(cosh(Df64::from(1000.0))));
        assert!(is_infinite(cosh(Df64::INFINITY)));
        assert!(is_infinite(cosh(Df64::NEG_INFINITY)));
        assert!(is_nan(cosh(Df64::NAN)));

        // simple vals
        check_unary(cosh, |x| x.cosh(), Df64::ZERO, 1.0);

        // small values
        let mut x = Df64::ONE;
        while x.hi > 1e-290 {
            check_unary(cosh, |x| x.cosh(), x, 1.0);
            check_unary(cosh, |x| x.cosh(), -x, 1.0);
            x *= 0.947;
        }

        // large values
        x = Df64::ONE;
        while x.hi < COSH_MAX {
            check_unary(cosh, |x| x.cosh(), x, 1.0);
            check_unary(cosh, |x| x.cosh(), -x, 1.0);
            x *= 1.0041;
        }

        assert!(is_finite(cosh(Df64::from(COSH_MAX))));
    }

    #[test]
    fn test_sinh() {
        // special values
        assert!(is_infinite(sinh(Df64::from(1000.0))));
        assert!(is_infinite(sinh(Df64::INFINITY)));
        assert!(is_infinite(sinh(Df64::NEG_INFINITY)));
        assert!(sinh(Df64::NEG_INFINITY) < Df64::ZERO);
        assert!(is_nan(sinh(Df64::NAN)));

        // simple vals
        check_unary(sinh, |x| x.sinh(), Df64::ZERO, 1.0);

        // small values
        let mut x = Df64::ONE;
        while x.hi > 1e-290 {
            check_unary(sinh, |x| x.sinh(), x, 2.0);
            check_unary(sinh, |x| x.sinh(), -x, 2.0);
            x *= 0.947;
        }

        // large values
        x = Df64::ONE;
        while x.hi < COSH_MAX {
            check_unary(sinh, |x| x.sinh(), x, 1.0);
            check_unary(sinh, |x| x.sinh(), -x, 1.0);
            x *= 1.0041;
        }

        assert!(is_finite(sinh(Df64::from(COSH_MAX))));
    }

    #[test]
    fn test_tanh() {
        // special values
        assert!(tanh(Df64::from(1000.0)) == Df64::ONE);
        assert!(tanh(Df64::from(-1000.0)) == Df64::from(-1.0));
        assert!(tanh(Df64::INFINITY) == Df64::ONE);
        assert!(tanh(-Df64::INFINITY) == Df64::from(-1.0));
        assert!(is_nan(tanh(Df64::NAN)));

        // simple vals
        check_unary(tanh, |x| x.tanh(), Df64::ZERO, 1.0);

        // small values
        let mut x = Df64::ONE;
        while x.hi > 1e-290 {
            check_unary(tanh, |x| x.tanh(), x, 2.0);
            check_unary(tanh, |x| x.tanh(), -x, 2.0);
            x *= 0.947;
        }

        // large values
        x = Df64::ONE;
        while x.hi < 10.0 * COSH_MAX {
            check_unary(tanh, |x| x.tanh(), x, 2.0);
            check_unary(tanh, |x| x.tanh(), -x, 2.0);
            x *= 1.0041;
        }
    }

    #[test]
    fn test_arc() {
        // small values
        let mut x = Df64::ONE;
        while x.hi > 1e-290 {
            check_unary(asinh, |x| x.asinh(), x, 2.0);
            check_unary(asinh, |x| x.asinh(), -x, 2.0);
            if x < Df64::ONE {
                check_unary(atanh, |x| x.atanh(), x, 2.5);
                check_unary(atanh, |x| x.atanh(), -x, 2.5);
            }
            x *= 0.91;
        }

        // large values
        // XXX not entire range covered
        x = Df64::ONE;
        while x.hi < f64::MAX / 4.0 {
            check_unary(asinh, |x| x.asinh(), x, 2.0);
            check_unary(asinh, |x| x.asinh(), -x, 2.0);
            check_unary(acosh, |x| x.acosh(), x, 2.0);
            x *= 1.13;
        }
    }
}
