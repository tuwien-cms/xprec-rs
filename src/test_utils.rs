use super::Df64;
use core::cmp::Ordering;
use rug::Assign;
use rug::Float;
use rug::float::Round;
use rug::ops::AssignRound;

pub const PREC: u32 = 120;

impl AssignRound<Df64> for Float {
    type Round = Round;
    type Ordering = Ordering;
    fn assign_round(&mut self, src: Df64, _round: Round) -> Ordering {
        let (hi, _hdir) = Float::with_val_round(PREC, src.hi, _round);
        let (lo, _ldir) = Float::with_val_round(PREC, src.lo, _round);
        *self = hi + lo;
        Ordering::Equal
    }
}

/// Checks the result of a unary function against a multiprecision result
///
///  * `f`     - function to be tested, called as `f(x)`
///  * `fref`  - expression on multifloats which produces the reference
///  * `x`     - argument
///  * `ulps`  - relative tolerance in the result as multiples of epsilon
///
pub fn check_unary<A: Copy>(f: fn(A) -> Df64, fref: fn(Float) -> Float, x: A, ulps: f64)
where
    Float: Assign<f64>,
    Float: Assign<A>,
{
    const EPSILON: f64 = Df64::EPSILON.hi;

    // Compute result to check
    let z = f(x);
    let zz = Float::with_val(PREC, z);

    // Compute reference result
    let xx = Float::with_val(PREC, x);
    let zz_ref = fref(xx);

    let diff = Float::with_val(PREC, &zz - &zz_ref);
    let thr = Float::with_val(PREC, EPSILON * ulps * zz.clone().abs());
    if !(diff.abs() <= thr) {
        // Recompute xx
        let xx = Float::with_val(PREC, x);
        let diff = Float::with_val(PREC, &zz - &zz_ref);
        let reldiff = Float::with_val(PREC, &diff / &zz);
        let ulpsdiff = Float::to_f64(&reldiff) / EPSILON;
        panic!(
            "f({})\n\t\
             should be   = {}\n\t\
             instead was = {}\n\t\
             deviation   = {:.3} ulps (exceeds threshold of {:.3} ulps)",
            xx, zz_ref, zz, ulpsdiff, ulps
        );
    }
}

/// Checks the result of a binary function against a multiprecision result
///
///  * `f`     - function to be tested, called as `f(x, y)`
///  * `fref`  - expression on multifloats which produces the reference
///  * `x`     - first argument
///  * `y`     - second argument
///  * `ulps`  - relative tolerance in the result as multiples of epsilon
///
pub fn check_binary<A: Copy, B: Copy>(
    f: fn(A, B) -> Df64,
    fref: fn(Float, Float) -> Float,
    x: A,
    y: B,
    ulps: f64,
) where
    Float: Assign<f64>,
    Float: Assign<A>,
    Float: Assign<B>,
{
    const EPSILON: f64 = Df64::EPSILON.hi;

    // Compute result to check
    let z = f(x, y);
    let zz = Float::with_val(PREC, z);

    // Compute reference result
    let xx = Float::with_val(PREC, x);
    let yy = Float::with_val(PREC, y);
    let zz_ref = fref(xx, yy);

    let diff = Float::with_val(PREC, &zz - &zz_ref);
    let thr = Float::with_val(PREC, EPSILON * ulps * zz.clone().abs());
    if !(diff.abs() <= thr) {
        // Recompute xx and yy
        let xx = Float::with_val(PREC, x);
        let yy = Float::with_val(PREC, y);
        let diff = Float::with_val(PREC, &zz - &zz_ref);
        let reldiff = Float::with_val(PREC, &diff / &zz);
        let ulpsdiff = Float::to_f64(&reldiff) / EPSILON;
        panic!(
            "f({}, {})\n\t\
             should be   = {}\n\t\
             instead was = {}\n\t\
             deviation   = {:.3} ulps (exceeds threshold of {:.3} ulps)",
            xx, yy, zz_ref, zz, ulpsdiff, ulps
        );
    }
}
