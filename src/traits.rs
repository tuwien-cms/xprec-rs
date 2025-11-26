/* Traits.
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::Df64;
use super::{
    AddFast, CompensatedArithmetic, CompensatedAdd, CompensatedSub,
    CompensatedMul, CompensatedDiv, CompensatedSqrt, SubFast};
use super::{arith, checks, circular, consts, exp, funcs, hyperbolic, roots, round};
use num_traits::{Inv, Num, One, Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf, SupersetOf};
use simba::simd::SimdValue;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

// ---------------------------------------------------------------------------
// STANDARD TRAITS

/// Macro for implementing binary operation traits
///
/// Implements traits `Trait` for a binary operation `func` for the following
/// combination of types
///
///   - `$op_qq: fn(Df64, Df64) -> Df64` ... `$Trait for Df64`
///   - `$op_qd: fn(Df64, f64) -> Df64` ... `$Trait<f64> for Df64`
///   - `$op_dq: fn(f64, Df64) -> Df64` ... `$Trait<Df64> for f64`
///
macro_rules! binary_op
{
    ($Trait:ident, $func:ident, $op_qq:path, $op_qd:path, $op_dq:path) => {
        // implementation for Df64 (op) Df64
        impl $Trait for Df64 {
            type Output = Df64;
            fn $func(self, b: Df64) -> Df64 {
                return $op_qq(self, b);
            }
        }
        // implementation for Df64 (op) f64
        impl $Trait<f64> for Df64 {
            type Output = Df64;
            fn $func(self, b: f64) -> Df64 {
                return $op_qd(self, b);
            }
        }
        // implementation for f64 (op) Df64
        impl $Trait<Df64> for f64 {
            type Output = Df64;
            fn $func(self, b: Df64) -> Df64 {
                return $op_dq(self, b);
            }
        }
    };
}

binary_op!(Add, add, arith::add_qq, arith::add_qd, arith::add_dq);
binary_op!(Sub, sub, arith::sub_qq, arith::sub_qd, arith::sub_dq);
binary_op!(Mul, mul, arith::mul_qq, arith::mul_qd, arith::mul_dq);
binary_op!(Div, div, arith::div_qq, arith::div_qd, arith::div_dq);
binary_op!(Rem, rem, round::mod_qq, round::mod_qd, round::mod_dq);

/// Macro for implementing in-place operation traits
///
/// Implements traits `Trait` for a inplace operation `func` for the following
/// combination of types
///
///   - `$op_qq: fn(Df64, Df64) -> Df64` ... `$Trait for Df64`
///   - `$op_qd: fn(Df64, f64) -> Df64` ... `$Trait<f64> for Df64`
///
macro_rules! inplace_op
{
    ($Trait:ident, $func:ident, $op_qq:expr, $op_qd:expr) => {
        impl $Trait for Df64 {
            fn $func(&mut self, other: Df64) {
                *self = $op_qq(*self, other);
            }
        }
        impl $Trait<f64> for Df64 {
            fn $func(&mut self, other: f64) {
                *self = $op_qd(*self, other);
            }
        }
    };
}

inplace_op!(AddAssign, add_assign, arith::add_qq, arith::add_qd);
inplace_op!(SubAssign, sub_assign, arith::sub_qq, arith::sub_qd);
inplace_op!(MulAssign, mul_assign, arith::mul_qq, arith::mul_qd);
inplace_op!(DivAssign, div_assign, arith::div_qq, arith::div_qd);
inplace_op!(RemAssign, rem_assign, round::mod_qq, round::mod_qd);

/// Macro for implementing unary operation traits
///
/// Implements traits `Trait` for a inplace operation `func` for the following
/// type:
///
///   - `$op_q: fn(Df64) -> Df64` ... `$Trait for Df64`
///
macro_rules! unary_op
{
    ($Trait:ident, $func:ident, $op_q:expr) => {
        impl $Trait for Df64 {
            type Output = Df64;
            fn $func(self) -> Df64 {
                return $op_q(self);
            }
        }
    }
}

unary_op!(Neg, neg, arith::neg_q);

// ---------------------------------------------------------------------------
// COMPENSATE

impl CompensatedArithmetic<f64> for Df64 {
    type Compensate = f64;

    #[inline(always)]
    fn compensate(self: &Df64) -> f64 {
        return self.lo;
    }
}

impl CompensatedAdd<f64> for Df64 {
    #[inline(always)]
    fn compensated_add(a: f64, b: f64) -> Df64 {
        return arith::add_dd(a, b);
    }

    #[inline(always)]
    fn compensated_fast_add(a: f64, b: f64) -> Df64 {
        return arith::addfast_dd(a, b);
    }
}

impl CompensatedSub<f64> for Df64 {
    #[inline(always)]
    fn compensated_sub(a: f64, b: f64) -> Df64 {
        return arith::sub_dd(a, b);
    }


    #[inline(always)]
    fn compensated_fast_sub(a: f64, b: f64) -> Df64 {
        return arith::subfast_dd(a, b);
    }
}

impl CompensatedMul<f64> for Df64 {
    #[inline(always)]
    fn compensated_mul(a: f64, b: f64) -> Df64 {
        return arith::mul_dd(a, b);
    }
}

impl CompensatedDiv<f64> for Df64 {
    #[inline(always)]
    fn compensated_div(a: f64, b: f64) -> Df64 {
        return arith::div_dd(a, b);
    }
}

impl CompensatedSqrt<f64> for Df64 {
    #[inline(always)]
    fn compensated_sqrt(a: f64) -> Df64 {
        return arith::sqrt_d(a);
    }
}

macro_rules! binary_op_fast
{
    ($Trait:ident, $func:ident, $op_qq:path, $op_qd:path, $op_dq:path) => {
        // implementation for Df64 (op) Df64
        impl $Trait for Df64 {
            fn $func(self, b: Df64) -> Df64 {
                return $op_qq(self, b);
            }
        }
        // implementation for Df64 (op) f64
        impl $Trait<f64> for Df64 {
            fn $func(self, b: f64) -> Df64 {
                return $op_qd(self, b);
            }
        }
        // implementation for f64 (op) Df64
        impl $Trait<Df64> for f64 {
            fn $func(self, b: Df64) -> Df64 {
                return $op_dq(self, b);
            }
        }
    };
}

binary_op_fast!(
    AddFast, add_fast, arith::addfast_qq, arith::addfast_qd, arith::addfast_dq);
binary_op_fast!(
    SubFast, sub_fast, arith::subfast_qq, arith::subfast_qd, arith::subfast_dq);

// ---------------------------------------------------------------------------
// NUMERIC TRAITS

impl Zero for Df64 {
    fn zero() -> Df64 {
        return Df64 {hi: 0.0, lo: 0.0};
    }
    fn is_zero(&self) -> bool {
        return self.hi == 0.0;
    }
}

impl One for Df64 {
    fn one() -> Df64 {
        return Df64 {hi: 1.0, lo: 0.0};
    }
    fn is_one(&self) -> bool {
        return self.hi == 1.0 && self.lo == 0.0;
    }
}

impl Inv for Df64 {
    type Output = Df64;
    fn inv(self) -> Df64 {
        return arith::reciprocal_q(self);
    }
}

impl Num for Df64 {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32)
            -> Result<Self, Self::FromStrRadixErr>
    {
        // XXX precision is insufficient
        let x64 = f64::from_str_radix(str, radix)?;
        return Ok(Df64::from(x64));
    }
}

impl std::fmt::Display for Df64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "(hi: {}, lo: {})", self.hi, self.lo)
    }
}

// XXX we use impl_primitive_simd_value_for_scalar! for now. Revisit.
impl SimdValue for Df64 {
    const LANES: usize = 1;
    type Element = Df64;
    type SimdBool = bool;

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        val
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        assert!(i < Self::LANES);
        unsafe {
            return self.extract_unchecked(i);
        }
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        assert!(i < Self::LANES);
        unsafe {
            self.replace_unchecked(i, val);
        }
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        return *self;
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        return if cond { self } else { other };
    }
}

impl Signed for Df64 {
    #[inline(always)]
    fn abs(&self) -> Self {
        return funcs::abs(*self);
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        return funcs::abs(arith::sub_qq(*self, *other));
    }

    #[inline(always)]
    fn signum(&self) -> Self {
        return Df64::from(self.hi.signum());
    }

    #[inline(always)]
    fn is_positive(&self) -> bool {
        return self.hi.is_sign_positive();
    }

    #[inline(always)]
    fn is_negative(&self) -> bool {
        return self.hi.is_sign_negative();
    }
}

impl SubsetOf<Self> for Df64 {
    #[inline(always)]
    fn to_superset(&self) -> Df64 {
        return *self;
    }

    #[inline(always)]
    fn from_superset_unchecked(element: &Df64) -> Df64 {
        return *element;
    }

    #[inline(always)]
    fn is_in_subset(_element: &Df64) -> bool {
        return true;
    }
}

macro_rules! impl_superset (
    ($subset:path as Df64) => {
        impl SupersetOf<$subset> for Df64 {
            #[inline(always)]
            fn is_in_subset(&self) -> bool {
                // Docs specify: The notion of “nested sets” is very broad and
                // applies to what the types are supposed to represent ... f32
                // and f64 are both supposed to represent reals and are thus
                // considered equal (even if in practice f64 has more elements)
                // This directly contradicts the FromPrimitive and ToPrimitive
                // traits.
                return true;
            }

            #[inline(always)]
            fn to_subset_unchecked(&self) -> $subset {
                return self.hi as $subset;
            }

            #[inline(always)]
            fn from_subset(element: &$subset) -> Self {
                // XXX remove .. as f64
                return Df64::from(*element as f64);
            }
        }
    }
);

impl_superset!(f64 as Df64);
impl_superset!(f32 as Df64);

impl Field for Df64 { }

impl ComplexField for Df64 {
    type RealField = Df64;

    #[inline(always)]
    fn from_real(re: Df64) -> Self {
        return re;
    }

    #[inline(always)]
    fn real(self) -> Df64 {
        return self;
    }

    #[inline(always)]
    fn imaginary(self) -> Df64 {
        return Df64::ZERO;
    }

    #[inline(always)]
    fn modulus(self) -> Df64 {
        return funcs::abs(self);
    }

    #[inline(always)]
    fn modulus_squared(self) -> Df64 {
        return arith::square_q(self);
    }

    #[inline]
    fn argument(self) -> Df64 {
        if self.hi.is_sign_negative() {
            return consts::PI;
        } else {
            return Df64::ZERO;
        }
    }

    #[inline(always)]
    fn norm1(self) -> Df64 {
        return funcs::abs(self);
    }

    #[inline(always)]
    fn scale(self, factor: Df64) -> Self {
        return self * factor;
    }

    #[inline(always)]
    fn unscale(self, factor: Df64) -> Self {
        return self / factor;
    }

    #[inline(always)]
    fn floor(self) -> Self {
        return round::floor(self);
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        return round::ceil(self);
    }

    #[inline(always)]
    fn round(self) -> Self {
        return round::round(self);
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        return round::trunc(self);
    }

    #[inline(always)]
    fn fract(self) -> Self {
        return funcs::fract(self);
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // There are two requirements that one has with fma: (1) it must be
        // accurate without intermediate rounding and (2) it must be at least
        // as fast as (a*b)+c. We have no way of satisfying both, so we go
        // for performance.
        return (self * a) + b;
    }

    #[inline(always)]
    fn abs(self) -> Df64 {
        return funcs::abs(self);
    }

    #[inline(always)]
    fn hypot(self,other:Self) -> Df64 {
        return roots::hypot(self, other);
    }

    #[inline(always)]
    fn recip(self) -> Self {
        return arith::reciprocal_q(self);
    }

    #[inline(always)]
    fn conjugate(self) -> Self {
        return self;
    }

    #[inline(always)]
    fn sin(self) -> Self {
        return circular::sin(self);
    }

    #[inline(always)]
    fn cos(self) -> Self {
        return circular::cos(self);
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self,Self) {
        return circular::sincos(self);
    }

    #[inline(always)]
    fn tan(self) -> Self {
        return circular::tan(self);
    }

    #[inline(always)]
    fn asin(self) -> Self {
        return circular::asin(self);
    }

    #[inline(always)]
    fn acos(self) -> Self {
        return circular::acos(self);
    }

    #[inline(always)]
    fn atan(self) -> Self {
        return circular::atan(self);
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        return hyperbolic::sinh(self);
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        return hyperbolic::cosh(self);
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        return hyperbolic::tanh(self);
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        return hyperbolic::asinh(self);
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        return hyperbolic::acosh(self);
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        return hyperbolic::atanh(self);
    }

    #[inline(always)]
    fn log(self, base:Df64) -> Self {
        return exp::log_base(self, base);
    }

    #[inline(always)]
    fn log2(self) -> Self {
        return exp::log2(self);
    }

    #[inline(always)]
    fn log10(self) -> Self {
        return exp::log10(self);
    }

    #[inline(always)]
    fn ln(self) -> Self {
        return exp::log(self);
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        return exp::log1p(self);
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        return arith::sqrt_q(self);
    }

    #[inline(always)]
    fn exp(self) -> Self {
        return exp::exp(self);
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        return exp::exp2(self);
    }

    #[inline(always)]
    fn exp_m1(self) -> Self {
        return exp::expm1(self);
    }

    #[inline(always)]
    fn powi(self,n:i32) -> Self {
        return exp::powi(self, n);
    }

    #[inline(always)]
    fn powf(self,n:Df64) -> Self {
        return exp::powf(self, n);
    }

    #[inline(always)]
    fn powc(self,n:Self) -> Self {
        return exp::powf(self, n);
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        return checks::is_finite(*self);
    }

    #[inline(always)]
    fn try_sqrt(self) -> Option<Self> {
        return Some(arith::sqrt_q(self));
    }
}

impl RealField for Df64 {
    fn is_sign_positive(&self) -> bool {
        return self.hi.is_sign_positive();
    }

    fn is_sign_negative(&self) -> bool {
        return self.hi.is_sign_negative();
    }

    fn copysign(self, sign: Self) -> Self {
        return funcs::copysign(self, sign);
    }

    fn max(self, other: Self) -> Self {
        return funcs::max(self, other);
    }

    fn min(self, other: Self) -> Self {
        return funcs::min(self, other);
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        return funcs::clamp(self, min, max);
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        return circular::atan2(self, other);
    }

    #[inline(always)]
    fn min_value() -> Option<Self> {
        return Some(Df64::MIN_POSITIVE);
    }

    #[inline(always)]
    fn max_value() -> Option<Self> {
        return Some(Df64::MAX);
    }

    /// Circle number π
    #[inline(always)]
    fn pi() -> Self {
        return consts::PI;
    }

    /// Two times π
    #[inline(always)]
    fn two_pi() -> Self {
        return consts::TWO_PI;
    }

    /// Half of π
    #[inline(always)]
    fn frac_pi_2() -> Self {
        return consts::PI_HALF;
    }

    /// One third of π
    #[inline(always)]
    fn frac_pi_3() -> Self {
        return consts::PI_THIRD;
    }

    /// One quarter of π
    #[inline(always)]
    fn frac_pi_4() -> Self {
        return consts::PI_FOURTH;
    }

    /// One sixth of π
    #[inline(always)]
    fn frac_pi_6() -> Self {
        return consts::PI_SIXTH;
    }

    /// One eighth of π
    #[inline(always)]
    fn frac_pi_8() -> Self {
        return consts::PI_EIGHTH;
    }

    /// Reciprocal of π
    #[inline(always)]
    fn frac_1_pi() -> Self {
        return consts::ONE_OVER_PI;
    }

    /// Twice the reciprocal of π
    #[inline(always)]
    fn frac_2_pi() -> Self {
        return consts::TWO_OVER_PI;
    }

    /// Twice the reciprocal of the square root of π
    #[inline(always)]
    fn frac_2_sqrt_pi() -> Self {
        return consts::TWO_OVER_SQRT_PI;
    }

    /// Euler number e
    #[inline(always)]
    fn e() -> Self {
        return consts::EULER_E;
    }

    /// Binary logarithm of e
    #[inline(always)]
    fn log2_e() -> Self {
        return consts::LOG2_E;
    }

    /// Logarithm base-10 of e
    #[inline(always)]
    fn log10_e() -> Self {
        return consts::LOG10_E;
    }

    /// Natural logarithm of 2
    #[inline(always)]
    fn ln_2() -> Self {
        return consts::LN_2;
    }

    /// Natural logarithm of 10
    #[inline(always)]
    fn ln_10() -> Self {
        return consts::LN_10;
    }
}

// ---------------------------------------------------------------------------
// UNIT TESTS

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_traits()
    {
        let x = Df64::ONE * 2.0;
        let y = Df64::ONE / 4.0;
        assert_eq!(1.0 + x * y - 2.0, Df64::from(-0.5));
    }

}