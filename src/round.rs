/* Implementations.
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::Df64;
use super::arith::{
    add_qd, addfast_dd, div_dq, div_qd, div_qq, mul_dq, mul_qq, subfast_dq, subfast_qq,
};

pub fn ceil(x: Df64) -> Df64 {
    // If hi was not an integer, it means that rounding it up/down/towards zero
    // already gives our answer.  This also covers NaN since then x != x.
    // We cannot simply truncate both hi and lo since they may have the same
    // sign
    let hi = x.hi.ceil();
    if hi != x.hi {
        return Df64::from(hi);
    }

    // hi is an integer, so modify lo instead.  This may actually increase the
    // magnitude above the limit, so let's renormalize to be safe.
    let lo = x.lo.ceil();
    return addfast_dd(hi, lo);
}

pub fn floor(x: Df64) -> Df64 {
    // If hi was not an integer, it means that rounding it up/down/towards zero
    // already gives our answer.  This also covers NaN since then x != x.
    // We cannot simply truncate both hi and lo since they may have the same
    // sign
    let hi = x.hi.floor();
    if hi != x.hi {
        return Df64::from(hi);
    }

    // hi is an integer, so modify lo instead.  This may actually increase the
    // magnitude above the limit, so let's renormalize to be safe.
    let lo = x.lo.floor();
    return addfast_dd(hi, lo);
}

pub fn trunc(x: Df64) -> Df64 {
    // If hi was not an integer, it means that rounding it up/down/towards zero
    // already gives our answer.  This also covers NaN since then x != x.
    // We cannot simply truncate both hi and lo since they may have opposite
    // signs.
    let hi = x.hi.trunc();
    if hi != x.hi {
        return Df64::from(hi);
    }

    // hi is an integer, so modify lo instead.  Here, one needs to be careful
    // to respect the truncation direction that hi requires, and so we have
    // to round towards -+infinity, for x > 0 and x < 0, repectively.
    //
    // This may actually increase the  magnitude above the limit, so let's
    // renormalize to be safe.
    let lo = if x.hi < 0.0 {
        x.lo.ceil()
    } else {
        x.lo.floor()
    };
    return addfast_dd(hi, lo);
}

pub fn round(x: Df64) -> Df64 {
    // trunc is fast, so it makes sense to use this as a building block.
    let nudge = (0.5_f64).copysign(x.hi);
    return trunc(add_qd(x, nudge));
}

pub fn mod_qq(x: Df64, y: Df64) -> Df64 {
    // XXX this loses an enormous amount of precision. Avoid.
    let i = trunc(div_qq(x, y));
    return subfast_qq(x, mul_qq(y, i));
}

pub fn mod_qd(x: Df64, y: f64) -> Df64 {
    // XXX this loses an enormous amount of precision. Avoid.
    let i = trunc(div_qd(x, y));
    return subfast_qq(x, mul_dq(y, i));
}

pub fn mod_dq(x: f64, y: Df64) -> Df64 {
    // XXX this loses an enormous amount of precision. Avoid.
    let i = trunc(div_dq(x, y));
    return subfast_dq(x, mul_qq(y, i));
}

// ---------------------------------------------------------------------------
// UNIT TESTS

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_trunc() {
        let u = 2e-32;

        assert!(trunc(Df64 { hi: 0.0, lo: 0.0 }) == Df64::ZERO);
        assert!(
            trunc(Df64 {
                hi: 2.5,
                lo: u / 4.0
            }) == Df64::from(2.0)
        );
        assert!(
            trunc(Df64 {
                hi: -2.3,
                lo: u / 4.0
            }) == Df64::from(-2.0)
        );

        assert!(
            trunc(Df64 {
                hi: 2.0,
                lo: u / 2.0
            }) == Df64::from(2.0)
        );
        assert!(
            trunc(Df64 {
                hi: 2.0,
                lo: -u / 2.0
            }) == Df64::ONE
        );
        assert!(
            trunc(Df64 {
                hi: -2.0,
                lo: u / 2.0
            }) == Df64::from(-1.0)
        );
        assert!(
            trunc(Df64 {
                hi: -2.0,
                lo: -u / 2.0
            }) == Df64::from(-2.0)
        );
    }

    #[test]
    fn test_ceil() {
        let u = 2e-32;

        assert!(ceil(Df64 { hi: 0.0, lo: 0.0 }) == Df64::ZERO);
        assert!(
            ceil(Df64 {
                hi: 2.5,
                lo: u / 4.0
            }) == Df64::from(3.0)
        );
        assert!(
            ceil(Df64 {
                hi: -2.3,
                lo: u / 4.0
            }) == Df64::from(-2.0)
        );

        assert!(
            ceil(Df64 {
                hi: 2.0,
                lo: u / 2.0
            }) == Df64::from(3.0)
        );
        assert!(
            ceil(Df64 {
                hi: 2.0,
                lo: -u / 2.0
            }) == Df64::from(2.0)
        );
        assert!(
            ceil(Df64 {
                hi: -2.0,
                lo: u / 2.0
            }) == Df64::from(-1.0)
        );
        assert!(
            ceil(Df64 {
                hi: -2.0,
                lo: -u / 2.0
            }) == Df64::from(-2.0)
        );
    }

    #[test]
    fn test_round() {
        let u = 2e-32;

        assert!(round(Df64 { hi: 0.0, lo: 0.0 }) == Df64::ZERO);
        assert!(
            round(Df64 {
                hi: 2.5,
                lo: u / 4.0
            }) == Df64::from(3.0)
        );
        assert!(round(Df64 { hi: 2.5, lo: 0.0 }) == Df64::from(3.0));
        assert!(
            round(Df64 {
                hi: 2.5,
                lo: -u / 3.0
            }) == Df64::from(2.0)
        );
        assert!(
            round(Df64 {
                hi: -2.3,
                lo: u / 4.0
            }) == Df64::from(-2.0)
        );
        assert!(round(Df64 { hi: -2.5, lo: 0.0 }) == Df64::from(-3.0));
        assert!(
            round(Df64 {
                hi: -2.5,
                lo: u / 4.0
            }) == Df64::from(-2.0)
        );
        assert!(
            round(Df64 {
                hi: -2.5,
                lo: -u / 4.0
            }) == Df64::from(-3.0)
        );

        assert!(
            round(Df64 {
                hi: 2.0,
                lo: u / 2.0
            }) == Df64::from(2.0)
        );
        assert!(
            round(Df64 {
                hi: 2.0,
                lo: -u / 2.0
            }) == Df64::from(2.0)
        );
        assert!(
            round(Df64 {
                hi: -2.0,
                lo: u / 2.0
            }) == Df64::from(-2.0)
        );
        assert!(
            round(Df64 {
                hi: -2.0,
                lo: -u / 2.0
            }) == Df64::from(-2.0)
        );
    }
}
