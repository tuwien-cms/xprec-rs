//! Inherent floating point methods for `Df64`.
//
// Copyright (C) 2023-2025 Markus Wallerberger and others
// SPDX-License-Identifier: MIT

use crate::{arith, checks, circular, consts, exp, funcs, hyperbolic, roots, round};
use crate::Df64;

/// Floating point operations available without importing a trait.
///
/// These methods mirror the inherent methods that `f64` provides, so that
/// `x.sqrt()`, `Df64::from(2.0).exp()` and friends can be called with nothing
/// but `use xprec::Df64;`.  The trait implementations for `Df64`
/// (`num_traits::Float`, `num_traits::Signed`, `simba::RealField`,
/// `simba::ComplexField`) forward to these methods, so generic and
/// non-generic code cannot drift apart.
///
/// Methods that are not implemented yet (`cbrt`, `integer_decode`) are
/// deliberately absent rather than forwarding to a `todo!()`.
impl Df64 {
    // ----- classification -------------------------------------------------

    /// Returns `true` if this value is `NaN` and `false` otherwise.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        return checks::is_nan(self);
    }

    /// Returns `true` if this value is positive infinity or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        return checks::is_infinite(self);
    }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        return checks::is_finite(self);
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal nor `NaN`.
    #[inline(always)]
    pub fn is_normal(self) -> bool {
        return checks::is_normal(self);
    }

    /// Returns `true` if the number is subnormal.
    #[inline(always)]
    pub fn is_subnormal(self) -> bool {
        return checks::is_subnormal(self);
    }

    /// Returns the floating point category of the number.
    #[inline(always)]
    pub fn classify(self) -> std::num::FpCategory {
        return checks::classify(self);
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0` and `NaN`.
    #[inline(always)]
    pub fn is_sign_positive(self) -> bool {
        return !checks::is_sign_negative(self);
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0` and `NaN`.
    #[inline(always)]
    pub fn is_sign_negative(self) -> bool {
        return checks::is_sign_negative(self);
    }

    // ----- basic arithmetic -----------------------------------------------

    /// Computes the absolute value of `self`.
    #[inline(always)]
    pub fn abs(self) -> Df64 {
        return funcs::abs(self);
    }

    /// Returns a number that represents the sign of `self`.
    #[inline(always)]
    pub fn signum(self) -> Df64 {
        return Df64::from(self.hi.signum());
    }

    /// Returns a number composed of the magnitude of `self` and the sign of `sign`.
    #[inline(always)]
    pub fn copysign(self, sign: Df64) -> Df64 {
        return funcs::copysign(self, sign);
    }

    /// Takes the reciprocal (inverse) of `self`, `1/self`.
    #[inline(always)]
    pub fn recip(self) -> Df64 {
        return arith::reciprocal_q(self);
    }

    /// Takes the square root of `self`.
    #[inline(always)]
    pub fn sqrt(self) -> Df64 {
        return arith::sqrt_q(self);
    }

    /// Fused multiply-add: computes `self * a + b` with one rounding.
    #[inline(always)]
    pub fn mul_add(self, a: Df64, b: Df64) -> Df64 {
        return arith::mul_add_qq(self, a, b);
    }

    // ----- rounding -------------------------------------------------------

    /// Returns the largest integer less than or equal to `self`.
    #[inline(always)]
    pub fn floor(self) -> Df64 {
        return round::floor(self);
    }

    /// Returns the smallest integer greater than or equal to `self`.
    #[inline(always)]
    pub fn ceil(self) -> Df64 {
        return round::ceil(self);
    }

    /// Returns the nearest integer to `self`, rounding half-way cases away from zero.
    #[inline(always)]
    pub fn round(self) -> Df64 {
        return round::round(self);
    }

    /// Returns the integer part of `self`, towards zero.
    #[inline(always)]
    pub fn trunc(self) -> Df64 {
        return round::trunc(self);
    }

    /// Returns the fractional part of `self`.
    #[inline(always)]
    pub fn fract(self) -> Df64 {
        return funcs::fract(self);
    }

    // ----- comparison -----------------------------------------------------

    /// Returns the minimum of the two numbers, ignoring `NaN`.
    #[inline(always)]
    pub fn min(self, other: Df64) -> Df64 {
        return funcs::min(self, other);
    }

    /// Returns the maximum of the two numbers, ignoring `NaN`.
    #[inline(always)]
    pub fn max(self, other: Df64) -> Df64 {
        return funcs::max(self, other);
    }

    /// Restricts a value to be within a specified range.
    #[inline(always)]
    pub fn clamp(self, min: Df64, max: Df64) -> Df64 {
        return funcs::clamp(self, min, max);
    }

    /// The positive difference of two numbers: `max(self - other, 0)`.
    #[inline(always)]
    pub fn abs_sub(self, other: Df64) -> Df64 {
        let diff = arith::sub_qq(self, other);
        if diff.hi < 0.0 {
            return Df64::ZERO;
        }
        return diff;
    }

    /// Computes the length of the hypotenuse of a right-angle triangle.
    #[inline(always)]
    pub fn hypot(self, other: Df64) -> Df64 {
        return roots::hypot(self, other);
    }

    // ----- exponential and logarithmic functions --------------------------

    /// Raises a number to an integer power.
    #[inline(always)]
    pub fn powi(self, n: i32) -> Df64 {
        return exp::powi(self, n);
    }

    /// Raises a number to a floating point power.
    #[inline(always)]
    pub fn powf(self, n: Df64) -> Df64 {
        return exp::powf(self, n);
    }

    /// Returns `e^(self)`, the exponential function.
    #[inline(always)]
    pub fn exp(self) -> Df64 {
        return exp::exp(self);
    }

    /// Returns `2^(self)`.
    #[inline(always)]
    pub fn exp2(self) -> Df64 {
        return exp::exp2(self);
    }

    /// Returns `e^(self) - 1` without intermediate rounding.
    #[inline(always)]
    pub fn exp_m1(self) -> Df64 {
        return exp::expm1(self);
    }

    /// Returns the natural logarithm of the number.
    #[inline(always)]
    pub fn ln(self) -> Df64 {
        return exp::log(self);
    }

    /// Returns `ln(1 + n)` without intermediate rounding.
    #[inline(always)]
    pub fn ln_1p(self) -> Df64 {
        return exp::log1p(self);
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    #[inline(always)]
    pub fn log(self, base: Df64) -> Df64 {
        return exp::log_base(self, base);
    }

    /// Returns the base 2 logarithm of the number.
    #[inline(always)]
    pub fn log2(self) -> Df64 {
        return exp::log2(self);
    }

    /// Returns the base 10 logarithm of the number.
    #[inline(always)]
    pub fn log10(self) -> Df64 {
        return exp::log10(self);
    }

    // ----- trigonometric functions ----------------------------------------

    /// Computes the sine of a number (in radians).
    #[inline(always)]
    pub fn sin(self) -> Df64 {
        return circular::sin(self);
    }

    /// Computes the cosine of a number (in radians).
    #[inline(always)]
    pub fn cos(self) -> Df64 {
        return circular::cos(self);
    }

    /// Computes the tangent of a number (in radians).
    #[inline(always)]
    pub fn tan(self) -> Df64 {
        return circular::tan(self);
    }

    /// Computes the arcsine of a number, in radians.
    #[inline(always)]
    pub fn asin(self) -> Df64 {
        return circular::asin(self);
    }

    /// Computes the arccosine of a number, in radians.
    #[inline(always)]
    pub fn acos(self) -> Df64 {
        return circular::acos(self);
    }

    /// Computes the arctangent of a number, in radians.
    #[inline(always)]
    pub fn atan(self) -> Df64 {
        return circular::atan(self);
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`).
    #[inline(always)]
    pub fn atan2(self, other: Df64) -> Df64 {
        return circular::atan2(self, other);
    }

    /// Simultaneously computes the sine and cosine of the number, returning
    /// `(sin(self), cos(self))`.
    #[inline(always)]
    pub fn sin_cos(self) -> (Df64, Df64) {
        return circular::sincos(self);
    }

    // ----- hyperbolic functions -------------------------------------------

    /// Hyperbolic sine function.
    #[inline(always)]
    pub fn sinh(self) -> Df64 {
        return hyperbolic::sinh(self);
    }

    /// Hyperbolic cosine function.
    #[inline(always)]
    pub fn cosh(self) -> Df64 {
        return hyperbolic::cosh(self);
    }

    /// Hyperbolic tangent function.
    #[inline(always)]
    pub fn tanh(self) -> Df64 {
        return hyperbolic::tanh(self);
    }

    /// Inverse hyperbolic sine function.
    #[inline(always)]
    pub fn asinh(self) -> Df64 {
        return hyperbolic::asinh(self);
    }

    /// Inverse hyperbolic cosine function.
    #[inline(always)]
    pub fn acosh(self) -> Df64 {
        return hyperbolic::acosh(self);
    }

    /// Inverse hyperbolic tangent function.
    #[inline(always)]
    pub fn atanh(self) -> Df64 {
        return hyperbolic::atanh(self);
    }

    // ----- angle conversion -----------------------------------------------

    /// Converts radians to degrees.
    #[inline(always)]
    pub fn to_degrees(self) -> Df64 {
        return arith::mul_qq(self, consts::DEGREES_PER_RADIAN);
    }

    /// Converts degrees to radians.
    #[inline(always)]
    pub fn to_radians(self) -> Df64 {
        return arith::mul_qq(self, consts::RADIANS_PER_DEGREE);
    }
}
