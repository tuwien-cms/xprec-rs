use crate::Df64;
use crate::{arith, checks};

#[inline]
pub fn ldexp(a: Df64, n: i32) -> Df64 {
    return Df64 {
        hi: libm::ldexp(a.hi, n),
        lo: libm::ldexp(a.lo, n),
    };
}

#[inline]
pub fn fract(x: Df64) -> Df64 {
    // The fractional part is simply the fractional part of both hi and lo.
    // In case x.hi is not integer, we have that fract(x.lo) is a true
    // compensate, and we could directly construct Df64 from the two parts.
    // However, if x.hi is integer, then its fractional part is zero, and
    // we have to renormalize, but can use fast addition because of the zero
    // hi part.
    return arith::addfast_dd(x.hi.fract(), x.lo.fract());
}

#[inline]
pub fn copysign(mag: Df64, sgn: Df64) -> Df64 {
    // The sign is determined by the hi part, however, the sign of hi and lo
    // need not be the same, so we cannot merely broadcast copysign to both
    // parts.
    if checks::is_sign_negative(mag) != checks::is_sign_negative(sgn) {
        arith::neg_q(mag)
    } else {
        mag
    }
}

#[inline]
pub fn abs(x: Df64) -> Df64 {
    if x.hi.is_sign_negative() {
        arith::neg_q(x)
    } else {
        x
    }
}

#[inline]
pub fn min(a: Df64, b: Df64) -> Df64 {
    // fmin considers NaN to be the largest number. (a <= b) is false with
    // either element being NaN, if a is NaN, then it is okay to return b;
    // but if b is NaN, we have to return a
    if a <= b || checks::is_nan(b) { a } else { b }
}

#[inline]
pub fn max(a: Df64, b: Df64) -> Df64 {
    if a <= b || checks::is_nan(a) { b } else { a }
}

pub fn clamp(a: Df64, min_: Df64, max_: Df64) -> Df64 {
    if min_ <= a && a <= max_ {
        return a;
    } else {
        assert!(min_ <= max_);
        if !(min_ <= a) {
            return min_;
        } else {
            return max_;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_fract() {
        let mut x = Df64::from(1e-30);
        while x.hi < 1e14 {
            let scale = x.hi.abs().max(1.0);
            check_unary(fract, |x| x.fract(), x, scale);
            check_unary(fract, |x| x.fract(), -x, scale);
            x *= 1.0141;
        }
    }
}
