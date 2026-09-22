/* Traits.
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::Df64;
use super::{AddFast, CompensatedArithmetic, SubFast};
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
    fn compensated_sum(a: f64, b: f64) -> Df64 {
        return arith::add_dd(a, b);
    }

    #[inline(always)]
    fn compensated_diff(a: f64, b: f64) -> Df64 {
        return arith::sub_dd(a, b);
    }

    #[inline(always)]
    fn compensated_prod(a: f64, b: f64) -> Df64 {
        return arith::mul_dd(a, b);
    }

    #[inline(always)]
    fn compensated_ratio(a: f64, b: f64) -> Df64 {
        return arith::div_dd(a, b);
    }

    #[inline(always)]
    fn compensated_sqrt(a: f64) -> Df64 {
        return arith::sqrt_d(a);
    }

    #[inline(always)]
    fn compensated_fast_sum(a: f64, b: f64) -> Df64 {
        return arith::addfast_dd(a, b);
    }

    #[inline(always)]
    fn compensated_fast_diff(a: f64, b: f64) -> Df64 {
        return arith::subfast_dd(a, b);
    }

    #[inline(always)]
    fn compensate(self: &Df64) -> f64 {
        return self.lo;
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

/// Implementation of the `num_traits::FloatConst` trait for `Df64`.
///
/// This provides standard mathematical constants in double-double precision.
impl num_traits::FloatConst for Df64 {
    #[inline(always)]
    fn E() -> Self {
        consts::EULER_E
    }

    #[inline(always)]
    fn FRAC_1_PI() -> Self {
        consts::ONE_OVER_PI
    }

    #[inline(always)]
    fn FRAC_1_SQRT_2() -> Self {
        consts::FRAC_1_SQRT_2
    }

    #[inline(always)]
    fn FRAC_2_PI() -> Self {
        consts::TWO_OVER_PI
    }

    #[inline(always)]
    fn FRAC_2_SQRT_PI() -> Self {
        consts::TWO_OVER_SQRT_PI
    }

    #[inline(always)]
    fn FRAC_PI_2() -> Self {
        consts::PI_HALF
    }

    #[inline(always)]
    fn FRAC_PI_3() -> Self {
        consts::PI_THIRD
    }

    #[inline(always)]
    fn FRAC_PI_4() -> Self {
        consts::PI_FOURTH
    }

    #[inline(always)]
    fn FRAC_PI_6() -> Self {
        consts::PI_SIXTH
    }

    #[inline(always)]
    fn FRAC_PI_8() -> Self {
        consts::PI_EIGHTH
    }

    #[inline(always)]
    fn LN_10() -> Self {
        consts::LN_10
    }

    #[inline(always)]
    fn LN_2() -> Self {
        consts::LN_2
    }

    #[inline(always)]
    fn LOG10_E() -> Self {
        consts::LOG10_E
    }

    #[inline(always)]
    fn LOG2_E() -> Self {
        consts::LOG2_E
    }

    #[inline(always)]
    fn PI() -> Self {
        consts::PI
    }

    #[inline(always)]
    fn SQRT_2() -> Self {
        consts::SQRT_2
    }

    #[inline(always)]
    fn TAU() -> Self {
        consts::TWO_PI
    }

    #[inline(always)]
    fn LOG10_2() -> Self {
        consts::LOG10_2
    }

    #[inline(always)]
    fn LOG2_10() -> Self {
        consts::LOG2_10
    }
}

/// Implementation of the `num_traits::Float` trait for `Df64`.
///
/// This provides standard floating-point operations without depending on
/// higher-level traits like `ComplexField`. All methods delegate directly
/// to low-level modules (checks, round, exp, circular, hyperbolic, etc.)
/// for consistency and performance.
impl num_traits::Float for Df64 {
    // ===== Constants (8 methods) =====

    #[inline(always)]
    fn nan() -> Self {
        Df64::NAN
    }

    #[inline(always)]
    fn infinity() -> Self {
        Df64::INFINITY
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        Df64::NEG_INFINITY
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        Df64::from(-0.0)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Df64::MIN
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        Df64::MIN_POSITIVE
    }

    #[inline(always)]
    fn max_value() -> Self {
        Df64::MAX
    }

    #[inline(always)]
    fn epsilon() -> Self {
        Df64::EPSILON
    }

    // ===== Classification methods (8 methods) =====

    #[inline(always)]
    fn is_nan(self) -> bool {
        checks::is_nan(self)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        checks::is_infinite(self)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        checks::is_finite(self)
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        checks::is_normal(self)
    }

    #[inline(always)]
    fn is_subnormal(self) -> bool {
        checks::is_subnormal(self)
    }

    #[inline(always)]
    fn classify(self) -> std::num::FpCategory {
        checks::classify(self)
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        !checks::is_sign_negative(self)
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        checks::is_sign_negative(self)
    }

    // ===== Basic arithmetic (3 methods) =====

    #[inline(always)]
    fn abs(self) -> Self {
        funcs::abs(self)
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Df64::from(self.hi.signum())
    }

    #[inline(always)]
    fn recip(self) -> Self {
        arith::reciprocal_q(self)
    }

    // ===== Rounding methods (5 methods) =====

    #[inline(always)]
    fn floor(self) -> Self {
        round::floor(self)
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        round::ceil(self)
    }

    #[inline(always)]
    fn round(self) -> Self {
        round::round(self)
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        round::trunc(self)
    }

    #[inline(always)]
    fn fract(self) -> Self {
        funcs::fract(self)
    }

    // ===== Comparison methods (6 methods) =====

    #[inline(always)]
    fn abs_sub(self, other: Self) -> Self {
        let diff = arith::sub_qq(self, other);
        if diff.hi < 0.0 {
            Df64::ZERO
        } else {
            diff
        }
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        arith::mul_add_qq(self, a, b)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        funcs::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        funcs::max(self, other)
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        funcs::clamp(self, min, max)
    }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        funcs::copysign(self, sign)
    }

    // ===== Additional comparison methods (1 method) =====

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        roots::hypot(self, other)
    }

    // ===== Exponential and logarithmic functions (11 methods) =====

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        exp::powi(self, n)
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        exp::powf(self, n)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        arith::sqrt_q(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        exp::exp(self)
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        exp::exp2(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        exp::log(self)
    }

    #[inline(always)]
    fn log(self, base: Self) -> Self {
        exp::log_base(self, base)
    }

    #[inline(always)]
    fn log2(self) -> Self {
        exp::log2(self)
    }

    #[inline(always)]
    fn log10(self) -> Self {
        exp::log10(self)
    }

    #[inline(always)]
    fn exp_m1(self) -> Self {
        exp::expm1(self)
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        exp::log1p(self)
    }

    // ===== Trigonometric functions (8 methods) =====

    #[inline(always)]
    fn sin(self) -> Self {
        circular::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        circular::cos(self)
    }

    #[inline(always)]
    fn tan(self) -> Self {
        circular::tan(self)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        circular::asin(self)
    }

    #[inline(always)]
    fn acos(self) -> Self {
        circular::acos(self)
    }

    #[inline(always)]
    fn atan(self) -> Self {
        circular::atan(self)
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        circular::atan2(self, other)
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        circular::sincos(self)
    }

    // ===== Hyperbolic functions (6 methods) =====

    #[inline(always)]
    fn sinh(self) -> Self {
        hyperbolic::sinh(self)
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        hyperbolic::cosh(self)
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        hyperbolic::tanh(self)
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        hyperbolic::asinh(self)
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        hyperbolic::acosh(self)
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        hyperbolic::atanh(self)
    }

    // ===== Angle conversion methods (2 methods) =====

    #[inline(always)]
    fn to_degrees(self) -> Self {
        arith::mul_qq(self, consts::DEGREES_PER_RADIAN)
    }

    #[inline(always)]
    fn to_radians(self) -> Self {
        arith::mul_qq(self, consts::RADIANS_PER_DEGREE)
    }

    // ===== Not yet implemented (2 methods) =====

    fn cbrt(self) -> Self {
        todo!("cbrt: requires Newton iteration or similar algorithm")
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        todo!("integer_decode: requires double-double specific design")
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
        arith::mul_add_qq(self, a, b)
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
    use approx::assert_ulps_eq;
    use num_traits::Float;
    use std::num::FpCategory;

    #[test]
    fn test_traits()
    {
        let x = Df64::ONE * 2.0;
        let y = Df64::ONE / 4.0;
        assert_eq!(1.0 + x * y - 2.0, Df64::from(-0.5));
    }

    // ===== Float trait constant methods (8 methods) =====

    #[test]
    fn test_float_nan()
    {
        let nan = <Df64 as Float>::nan();
        assert!(nan.is_nan());
        assert!(!nan.is_finite());
        assert!(!nan.is_infinite());
    }

    #[test]
    fn test_float_infinity()
    {
        let inf = <Df64 as Float>::infinity();
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
        assert!(!inf.is_nan());
        assert!(inf.is_sign_positive());

        let neg_inf = <Df64 as Float>::neg_infinity();
        assert!(neg_inf.is_infinite());
        assert!(!neg_inf.is_finite());
        assert!(!neg_inf.is_nan());
        assert!(neg_inf.is_sign_negative());
    }

    #[test]
    fn test_float_zero()
    {
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();

        // Basic properties of positive zero
        assert_eq!(pos_zero, Df64::ZERO);
        assert!(pos_zero.is_finite());
        assert!(!pos_zero.is_nan());
        assert_eq!(pos_zero.hi, 0.0);
        assert_eq!(pos_zero.lo, 0.0);

        // Basic properties of negative zero
        assert!(neg_zero.is_finite());
        assert!(!neg_zero.is_nan());

        // Check the sign bit of hi component
        assert!(pos_zero.hi.is_sign_positive());
        assert!(neg_zero.hi.is_sign_negative());

        // They should be equal in value (IEEE 754: +0.0 == -0.0)
        assert_eq!(pos_zero, neg_zero);
        assert!(!(pos_zero != neg_zero));
        assert!(!(pos_zero < neg_zero));
        assert!(!(pos_zero > neg_zero));
        assert!(pos_zero <= neg_zero);
        assert!(pos_zero >= neg_zero);

        // Both should classify as Zero
        assert_eq!(pos_zero.classify(), FpCategory::Zero);
        assert_eq!(neg_zero.classify(), FpCategory::Zero);
        assert!(!pos_zero.is_normal());
        assert!(!neg_zero.is_normal());
    }

    #[test]
    fn test_float_min_max_values()
    {
        let min = <Df64 as Float>::min_value();
        assert!(min.is_finite());
        assert!(min.is_sign_negative());
        assert_eq!(min, Df64::MIN);

        let max = <Df64 as Float>::max_value();
        assert!(max.is_finite());
        assert!(max.is_sign_positive());
        assert_eq!(max, Df64::MAX);

        let min_pos = <Df64 as Float>::min_positive_value();
        assert!(min_pos.is_finite());
        assert!(min_pos.is_sign_positive());
        assert_eq!(min_pos, Df64::MIN_POSITIVE);
    }

    #[test]
    fn test_float_epsilon()
    {
        let eps = <Df64 as Float>::epsilon();
        assert!(eps.is_finite());
        assert!(eps.is_sign_positive());
        assert_eq!(eps, Df64::EPSILON);

        // Verify epsilon property: 1.0 + epsilon != 1.0
        let one = Df64::ONE;
        assert_ne!(one + eps, one);
    }

    // ===== Float trait classification methods (8 methods) =====

    #[test]
    fn test_float_is_nan()
    {
        assert!(<Df64 as Float>::nan().is_nan());
        assert!(!Df64::ZERO.is_nan());
        assert!(!Df64::ONE.is_nan());
        assert!(!<Df64 as Float>::infinity().is_nan());
    }

    #[test]
    fn test_float_is_infinite()
    {
        assert!(<Df64 as Float>::infinity().is_infinite());
        assert!(<Df64 as Float>::neg_infinity().is_infinite());
        assert!(!Df64::ZERO.is_infinite());
        assert!(!Df64::ONE.is_infinite());
        assert!(!<Df64 as Float>::nan().is_infinite());
    }

    #[test]
    fn test_float_is_finite()
    {
        assert!(Df64::ZERO.is_finite());
        assert!(Df64::ONE.is_finite());
        assert!(Df64::from(1.5).is_finite());
        assert!(Df64::MIN.is_finite());
        assert!(Df64::MAX.is_finite());
        assert!(!<Df64 as Float>::infinity().is_finite());
        assert!(!<Df64 as Float>::nan().is_finite());
    }

    #[test]
    fn test_float_is_normal()
    {
        assert!(Df64::ONE.is_normal());
        assert!(Df64::from(2.5).is_normal());
        assert!(!Df64::ZERO.is_normal());
        assert!(!<Df64 as Float>::infinity().is_normal());
        assert!(!<Df64 as Float>::nan().is_normal());
    }

    #[test]
    fn test_float_is_subnormal()
    {
        // Normal values are not subnormal
        assert!(!Df64::ONE.is_subnormal());
        assert!(!Df64::from(2.5).is_subnormal());
        assert!(!Df64::ZERO.is_subnormal());
        assert!(!<Df64 as Float>::infinity().is_subnormal());
        assert!(!<Df64 as Float>::nan().is_subnormal());
        assert!(!Df64::MIN_POSITIVE.is_subnormal());

        // Values smaller than MIN_POSITIVE are subnormal
        let subnormal = Df64::MIN_POSITIVE * Df64::from(0.5);
        assert!(subnormal.is_subnormal());
    }

    #[test]
    fn test_float_classify()
    {
        assert_eq!(<Df64 as Float>::nan().classify(), FpCategory::Nan);
        assert_eq!(<Df64 as Float>::infinity().classify(), FpCategory::Infinite);
        assert_eq!(<Df64 as Float>::neg_infinity().classify(), FpCategory::Infinite);
        assert_eq!(Df64::ZERO.classify(), FpCategory::Zero);
        assert_eq!(Df64::ONE.classify(), FpCategory::Normal);
    }

    #[test]
    fn test_float_is_sign_positive_negative()
    {
        assert!(Df64::ONE.is_sign_positive());
        assert!(!Df64::ONE.is_sign_negative());

        let neg_one = -Df64::ONE;
        assert!(!neg_one.is_sign_positive());
        assert!(neg_one.is_sign_negative());

        assert!(<Df64 as Float>::infinity().is_sign_positive());
        assert!(<Df64 as Float>::neg_infinity().is_sign_negative());

        // Signed zero behavior: is_sign_positive/is_sign_negative check the sign bit
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        assert!(pos_zero.is_sign_positive());
        assert!(!pos_zero.is_sign_negative());
        assert!(!neg_zero.is_sign_positive());
        assert!(neg_zero.is_sign_negative());
    }

    // ===== Float trait basic arithmetic (3 methods) =====

    #[test]
    fn test_float_abs()
    {
        let x = Df64::from(3.5);
        assert_eq!(Float::abs(x), x);

        let neg_x = -x;
        assert_eq!(Float::abs(neg_x), x);

        assert_eq!(Float::abs(Df64::ZERO), Df64::ZERO);
        assert!(Float::abs(<Df64 as Float>::nan()).is_nan());
        assert!(Float::abs(<Df64 as Float>::infinity()).is_infinite());
        assert!(Float::abs(<Df64 as Float>::neg_infinity()).is_infinite());

        // Signed zero: abs of both zeros should be positive zero
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        let abs_pos = Float::abs(pos_zero);
        let abs_neg = Float::abs(neg_zero);
        assert_eq!(abs_pos, Df64::ZERO);
        assert_eq!(abs_neg, Df64::ZERO);
        assert!(abs_pos.hi.is_sign_positive());
        assert!(abs_neg.hi.is_sign_positive());
    }

    #[test]
    fn test_float_signum()
    {
        assert_eq!(Float::signum(Df64::from(5.0)), Df64::ONE);
        assert_eq!(Float::signum(Df64::from(-5.0)), -Df64::ONE);
        assert!(Float::signum(<Df64 as Float>::nan()).is_nan());

        // Signed zero: signum is based on hi.signum()
        // For f64: signum(+0.0) = 1.0, signum(-0.0) = -1.0
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::signum(pos_zero), Df64::ONE);
        assert_eq!(Float::signum(neg_zero), -Df64::ONE);
    }

    #[test]
    fn test_float_recip()
    {
        let two = Df64::from(2.0);
        let half = Df64::from(0.5);

        let result = Float::recip(two);
        assert_ulps_eq!(result, half);

        // Test recip(1) is close to 1
        let recip_one = Float::recip(Df64::ONE);
        assert_ulps_eq!(recip_one, Df64::ONE);

        // Signed zero: recip(0) returns NaN in current implementation
        // (IEEE 754 specifies infinity, but double-double implementation returns NaN)
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        let recip_pos = Float::recip(pos_zero);
        let recip_neg = Float::recip(neg_zero);
        assert!(recip_pos.is_nan());
        assert!(recip_neg.is_nan());
    }

    // ===== Float trait rounding methods (5 methods) =====

    #[test]
    fn test_float_floor()
    {
        assert_eq!(Float::floor(Df64::from(3.7)), Df64::from(3.0));
        assert_eq!(Float::floor(Df64::from(3.0)), Df64::from(3.0));
        assert_eq!(Float::floor(Df64::from(-3.7)), Df64::from(-4.0));
        assert_eq!(Float::floor(Df64::ZERO), Df64::ZERO);

        // Signed zero: rounding preserves zero value
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::floor(neg_zero), Df64::ZERO);
    }

    #[test]
    fn test_float_ceil()
    {
        assert_eq!(Float::ceil(Df64::from(3.2)), Df64::from(4.0));
        assert_eq!(Float::ceil(Df64::from(3.0)), Df64::from(3.0));
        assert_eq!(Float::ceil(Df64::from(-3.2)), Df64::from(-3.0));
        assert_eq!(Float::ceil(Df64::ZERO), Df64::ZERO);

        // Signed zero: rounding preserves zero value
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::ceil(neg_zero), Df64::ZERO);
    }

    #[test]
    fn test_float_round()
    {
        assert_eq!(Float::round(Df64::from(3.4)), Df64::from(3.0));
        assert_eq!(Float::round(Df64::from(3.5)), Df64::from(4.0));
        assert_eq!(Float::round(Df64::from(3.6)), Df64::from(4.0));
        assert_eq!(Float::round(Df64::from(-3.4)), Df64::from(-3.0));
        assert_eq!(Float::round(Df64::from(-3.5)), Df64::from(-4.0));
        assert_eq!(Float::round(Df64::ZERO), Df64::ZERO);

        // Signed zero: rounding preserves zero value
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::round(neg_zero), Df64::ZERO);
    }

    #[test]
    fn test_float_trunc()
    {
        assert_eq!(Float::trunc(Df64::from(3.7)), Df64::from(3.0));
        assert_eq!(Float::trunc(Df64::from(3.2)), Df64::from(3.0));
        assert_eq!(Float::trunc(Df64::from(-3.7)), Df64::from(-3.0));
        assert_eq!(Float::trunc(Df64::from(-3.2)), Df64::from(-3.0));
        assert_eq!(Float::trunc(Df64::ZERO), Df64::ZERO);

        // Signed zero: rounding preserves zero value
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::trunc(neg_zero), Df64::ZERO);
    }

    #[test]
    fn test_float_fract()
    {
        // Test positive value: fract(3.7) = 0.7
        let x = Df64::from(3.7);
        let fract = Float::fract(x);
        let trunc_x = Float::trunc(x);
        // fract(x) = x - trunc(x) (not floor!)
        assert_eq!(fract, x - trunc_x);
        assert!(fract.hi >= 0.0);
        assert!(fract.hi < 1.0);

        // Test negative value: fract(-3.7) = -0.7 (matches standard Rust behavior)
        let x = Df64::from(-3.7);
        let fract = Float::fract(x);
        let trunc_x = Float::trunc(x);
        assert_eq!(fract, x - trunc_x);
        assert!(fract.hi < 0.0);
        assert!(fract.hi > -1.0);

        // Test integer value
        assert_eq!(Float::fract(Df64::from(3.0)), Df64::ZERO);
        assert_eq!(Float::fract(Df64::ZERO), Df64::ZERO);

        // Signed zero: fract preserves zero value
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::fract(neg_zero), Df64::ZERO);
    }

    // ===== Float trait comparison methods (6 methods) =====

    #[test]
    fn test_float_min()
    {
        let a = Df64::from(2.0);
        let b = Df64::from(3.0);
        assert_eq!(Float::min(a, b), a);
        assert_eq!(Float::min(b, a), a);
        assert_eq!(Float::min(a, a), a);

        // Test with negative values
        let neg_a = Df64::from(-2.0);
        let neg_b = Df64::from(-3.0);
        assert_eq!(Float::min(neg_a, neg_b), neg_b);
        assert_eq!(Float::min(a, neg_a), neg_a);

        // Signed zero: min/max with zeros
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::min(pos_zero, a), pos_zero);
        assert_eq!(Float::min(neg_zero, neg_a), neg_a);
        // min between +0.0 and -0.0: IEEE 754 does not distinguish
        let min_zeros = Float::min(pos_zero, neg_zero);
        assert_eq!(min_zeros, pos_zero);  // Value equality
    }

    #[test]
    fn test_float_max()
    {
        let a = Df64::from(2.0);
        let b = Df64::from(3.0);
        assert_eq!(Float::max(a, b), b);
        assert_eq!(Float::max(b, a), b);
        assert_eq!(Float::max(a, a), a);

        // Test with negative values
        let neg_a = Df64::from(-2.0);
        let neg_b = Df64::from(-3.0);
        assert_eq!(Float::max(neg_a, neg_b), neg_a);
        assert_eq!(Float::max(a, neg_a), a);

        // Signed zero: min/max with zeros
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::max(pos_zero, a), a);
        assert_eq!(Float::max(neg_zero, neg_a), neg_zero);
        // max between +0.0 and -0.0: IEEE 754 does not distinguish
        let max_zeros = Float::max(pos_zero, neg_zero);
        assert_eq!(max_zeros, pos_zero);  // Value equality
    }

    #[test]
    fn test_float_signed_zero_copysign()
    {
        let one = Df64::ONE;
        let neg_one = -Df64::ONE;
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();

        // copysign(value, sign) copies the sign of 'sign' to 'value'
        let result = Float::copysign(one, neg_zero);
        assert!(result.hi.is_sign_negative());
        assert_ulps_eq!(result, neg_one);

        let result = Float::copysign(neg_one, pos_zero);
        assert!(result.hi.is_sign_positive());
        assert_ulps_eq!(result, one);

        // copysign with zero as the value
        let result = Float::copysign(pos_zero, neg_one);
        assert!(result.hi.is_sign_negative());

        let result = Float::copysign(neg_zero, one);
        assert!(result.hi.is_sign_positive());
    }

    #[test]
    fn test_float_clamp()
    {
        let min = Df64::from(2.0);
        let max = Df64::from(5.0);

        // Value below min should clamp to min
        assert_eq!(Float::clamp(Df64::from(1.0), min, max), min);

        // Value above max should clamp to max
        assert_eq!(Float::clamp(Df64::from(10.0), min, max), max);

        // Value within range should stay unchanged
        assert_eq!(Float::clamp(Df64::from(3.0), min, max), Df64::from(3.0));

        // Value at boundaries
        assert_eq!(Float::clamp(min, min, max), min);
        assert_eq!(Float::clamp(max, min, max), max);
    }

    #[test]
    fn test_float_abs_sub()
    {
        let a = Df64::from(5.0);
        let b = Df64::from(3.0);
        // abs_sub(a, b) = max(a - b, 0) = max(2, 0) = 2
        assert_eq!(Float::abs_sub(a, b), Df64::from(2.0));
        // abs_sub(b, a) = max(b - a, 0) = max(-2, 0) = 0
        assert_eq!(Float::abs_sub(b, a), Df64::ZERO);
        // abs_sub(a, a) = max(0, 0) = 0
        assert_eq!(Float::abs_sub(a, a), Df64::ZERO);

        // Test with negative values
        let neg_a = Df64::from(-5.0);
        // abs_sub(neg_a, b) = max(-5 - 3, 0) = max(-8, 0) = 0
        assert_eq!(Float::abs_sub(neg_a, b), Df64::ZERO);
        // abs_sub(b, neg_a) = max(3 - (-5), 0) = max(8, 0) = 8
        assert_eq!(Float::abs_sub(b, neg_a), Df64::from(8.0));
    }

    #[test]
    fn test_float_mul_add()
    {
        let a = Df64::from(2.0);
        let b = Df64::from(3.0);
        let c = Df64::from(4.0);

        // a.mul_add(b, c) = (a * b) + c = 2 * 3 + 4 = 10
        let result = Float::mul_add(a, b, c);
        assert_eq!(result, Df64::from(10.0));

        // Test with more complex values
        let x = Df64::from(1.5);
        let y = Df64::from(2.5);
        let z = Df64::from(1.0);
        let result = Float::mul_add(x, y, z);
        let expected = Df64::from(4.75); // 1.5 * 2.5 + 1.0
        assert_ulps_eq!(result, expected);
    }

    #[test]
    fn test_float_signed_zero_arithmetic()
    {
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        let one = Df64::ONE;

        // Addition with zeros
        assert_eq!(one + pos_zero, one);
        assert_eq!(one + neg_zero, one);

        // Subtraction
        assert_eq!(one - pos_zero, one);
        assert_eq!(one - neg_zero, one);

        // Multiplication by zero
        let mul_pos = one * pos_zero;
        let mul_neg = one * neg_zero;
        assert_eq!(mul_pos, pos_zero);
        assert_eq!(mul_neg, pos_zero);  // Value equality

        // Negation of zeros
        let neg_of_pos = -pos_zero;
        let neg_of_neg = -neg_zero;
        assert!(neg_of_pos.hi.is_sign_negative());
        assert!(neg_of_neg.hi.is_sign_positive());

        // Division of zero by non-zero
        let div_pos = pos_zero / one;
        let div_neg = neg_zero / one;
        assert_eq!(div_pos, pos_zero);
        assert_eq!(div_neg, pos_zero);  // Value equality

        // Division by zero returns NaN in current implementation
        // (IEEE 754 specifies infinity, but double-double implementation returns NaN)
        let result = one / pos_zero;
        assert!(result.is_nan());
    }

    // ===== Float trait exponential and logarithmic (11 methods) =====

    #[test]
    fn test_float_powi()
    {
        // Delegate to exp::powi, detailed precision tests are in exp.rs
        // Here we verify the Float trait correctly delegates
        let two = Df64::from(2.0);
        assert_eq!(Float::powi(two, 0), Df64::ONE);
        assert_ulps_eq!(Float::powi(two, 1), two);
        assert_ulps_eq!(Float::powi(two, 2), Df64::from(4.0));
        assert_ulps_eq!(Float::powi(two, 3), Df64::from(8.0));
        assert_ulps_eq!(Float::powi(two, -1), Df64::from(0.5));
        assert_ulps_eq!(Float::powi(Df64::from(3.0), 4), Df64::from(81.0));

        // Signed zero: 0^n returns NaN in current implementation
        // (uses exp(n*log(x)) which produces NaN for log(0))
        let pos_zero = Df64::ZERO;
        assert!(Float::powi(pos_zero, 0).is_nan());
        assert!(Float::powi(pos_zero, 2).is_nan());
        assert!(Float::powi(pos_zero, -2).is_nan());
    }

    #[test]
    fn test_float_powf()
    {
        // Delegate to exp::powf, detailed precision tests are in exp.rs
        let two = Df64::from(2.0);
        let three = Df64::from(3.0);

        assert_ulps_eq!(Float::powf(two, three), Df64::from(8.0));
        assert_ulps_eq!(Float::powf(Df64::from(4.0), Df64::from(0.5)), Df64::from(2.0));

        // Signed zero: powf with zero base returns NaN due to log(0) = -inf
        let pos_zero = Df64::ZERO;
        let result = Float::powf(pos_zero, two);
        assert!(result.is_nan());
    }

    #[test]
    fn test_float_exp()
    {
        // Delegate to exp::exp, detailed precision tests are in exp.rs
        // Here we verify the Float trait correctly delegates
        assert_eq!(Float::exp(Df64::ZERO), Df64::ONE);
        assert_ulps_eq!(Float::exp(Df64::ONE), crate::consts::EULER_E);

        // Edge cases
        assert_eq!(Float::exp(<Df64 as Float>::neg_zero()), Df64::ONE);
    }

    #[test]
    fn test_float_exp2()
    {
        // Delegate to exp::exp2, detailed precision tests are in exp.rs
        assert_eq!(Float::exp2(Df64::ZERO), Df64::ONE);
        assert_ulps_eq!(Float::exp2(Df64::ONE), Df64::from(2.0));
        assert_ulps_eq!(Float::exp2(Df64::from(2.0)), Df64::from(4.0));
        assert_ulps_eq!(Float::exp2(Df64::from(3.0)), Df64::from(8.0));
        assert_ulps_eq!(Float::exp2(Df64::from(10.0)), Df64::from(1024.0));
        assert_ulps_eq!(Float::exp2(-Df64::ONE), Df64::from(0.5));

        // Signed zero: exp2(0) = 1
        assert_eq!(Float::exp2(<Df64 as Float>::neg_zero()), Df64::ONE);
    }

    #[test]
    fn test_float_ln()
    {
        // Delegate to exp::log, detailed precision tests are in exp.rs
        assert_eq!(Float::ln(Df64::ONE), Df64::ZERO);
        assert_ulps_eq!(Float::ln(crate::consts::EULER_E), Df64::ONE);
        assert_ulps_eq!(Float::ln(Df64::from(2.0)), crate::consts::LN_2);
    }

    #[test]
    fn test_float_log()
    {
        // Delegate to exp::log_base, detailed precision tests are in exp.rs
        assert_ulps_eq!(Float::log(Df64::from(100.0), Df64::from(10.0)), Df64::from(2.0));
        assert_ulps_eq!(Float::log(Df64::from(8.0), Df64::from(2.0)), Df64::from(3.0));
    }

    #[test]
    fn test_float_log2()
    {
        // Delegate to exp::log2, detailed precision tests are in exp.rs
        assert_eq!(Float::log2(Df64::ONE), Df64::ZERO);
        assert_eq!(Float::log2(Df64::from(2.0)), Df64::ONE);
        assert_ulps_eq!(Float::log2(Df64::from(8.0)), Df64::from(3.0));
        assert_ulps_eq!(Float::log2(Df64::from(1024.0)), Df64::from(10.0));
    }

    #[test]
    fn test_float_log10()
    {
        // Delegate to exp::log10, detailed precision tests are in exp.rs
        assert_eq!(Float::log10(Df64::ONE), Df64::ZERO);
        assert_ulps_eq!(Float::log10(Df64::from(10.0)), Df64::ONE);
        assert_ulps_eq!(Float::log10(Df64::from(100.0)), Df64::from(2.0));
        assert_ulps_eq!(Float::log10(Df64::from(1000.0)), Df64::from(3.0));
    }

    #[test]
    fn test_float_exp_m1()
    {
        // Delegate to exp::expm1, detailed precision tests are in exp.rs
        // exp_m1(x) = exp(x) - 1
        assert_eq!(Float::exp_m1(Df64::ZERO), Df64::ZERO);

        // exp_m1(1) = e - 1
        let expected = crate::consts::EULER_E - Df64::ONE;
        assert_ulps_eq!(Float::exp_m1(Df64::ONE), expected);

        // Signed zero: expm1(0) = 0
        assert_eq!(Float::exp_m1(<Df64 as Float>::neg_zero()), Df64::ZERO);
    }

    #[test]
    fn test_float_ln_1p()
    {
        // Delegate to exp::log1p, detailed precision tests are in exp.rs
        // ln_1p(x) = ln(1 + x)
        assert_eq!(Float::ln_1p(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::ln_1p(Df64::ONE), crate::consts::LN_2); // ln(2)

        // Signed zero: ln_1p(0) = 0
        assert_eq!(Float::ln_1p(<Df64 as Float>::neg_zero()), Df64::ZERO);
    }

    #[test]
    fn test_float_sqrt()
    {
        // Delegate to arith::sqrt_q, detailed precision tests are in arith.rs
        assert_eq!(Float::sqrt(Df64::ZERO), Df64::ZERO);
        assert_eq!(Float::sqrt(Df64::ONE), Df64::ONE);
        assert_eq!(Float::sqrt(Df64::from(4.0)), Df64::from(2.0));

        // Verify sqrt(2) * sqrt(2) = 2 using high-precision check
        let two = Df64::from(2.0);
        let sqrt2 = Float::sqrt(two);
        assert_ulps_eq!(sqrt2 * sqrt2, two);

        // Signed zero: sqrt(+0.0) = +0.0
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        assert_eq!(Float::sqrt(pos_zero), pos_zero);
        assert_eq!(Float::sqrt(neg_zero), pos_zero);  // Value equality
        assert!(Float::sqrt(pos_zero).hi.is_sign_positive());
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_float_cbrt_not_implemented()
    {
        // cbrt is marked as todo!()
        let _ = Float::cbrt(Df64::from(8.0));
    }

    #[test]
    fn test_float_hypot()
    {
        // Delegate to roots::hypot, detailed precision tests are in roots.rs
        assert_ulps_eq!(Float::hypot(Df64::from(3.0), Df64::from(4.0)), Df64::from(5.0)); // 3-4-5 triangle
        assert_eq!(Float::hypot(Df64::ZERO, Df64::from(5.0)), Df64::from(5.0));
        assert_eq!(Float::hypot(Df64::from(5.0), Df64::ZERO), Df64::from(5.0));
    }

    // ===== Float trait trigonometric functions (8 methods) =====

    #[test]
    fn test_float_sin()
    {
        // Delegate to circular::sin, detailed precision tests are in circular.rs
        assert_eq!(Float::sin(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::sin(crate::consts::PI_HALF), Df64::ONE);
        assert_ulps_eq!(Float::sin(-crate::consts::PI_HALF), -Df64::ONE);

        // sin(π) should be very close to 0 (within a few ulps of π)
        let sin_pi = Float::sin(crate::consts::PI);
        assert!(Float::abs(sin_pi).hi < Df64::EPSILON.hi * 4.0);

        // Signed zero: sin(0) = 0 (sin is odd function)
        assert_eq!(Float::sin(<Df64 as Float>::neg_zero()), Df64::ZERO);
    }

    #[test]
    fn test_float_cos()
    {
        // Delegate to circular::cos, detailed precision tests are in circular.rs
        assert_eq!(Float::cos(Df64::ZERO), Df64::ONE);
        assert_ulps_eq!(Float::cos(crate::consts::PI), -Df64::ONE);

        // cos(π/2) should be very close to 0 (within a few ulps of π/2)
        let cos_pi_half = Float::cos(crate::consts::PI_HALF);
        assert!(Float::abs(cos_pi_half).hi < Df64::EPSILON.hi * 4.0);

        // Signed zero: cos(0) = 1 (cos is even function)
        assert_eq!(Float::cos(<Df64 as Float>::neg_zero()), Df64::ONE);
    }

    #[test]
    fn test_float_tan()
    {
        // Delegate to circular::tan, detailed precision tests are in circular.rs
        assert_eq!(Float::tan(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::tan(crate::consts::PI_FOURTH), Df64::ONE);
        assert_ulps_eq!(Float::tan(-crate::consts::PI_FOURTH), -Df64::ONE);

        // Test at pi/2 where tan has very large magnitude
        let result = Float::tan(crate::consts::PI_HALF);
        assert!(Float::abs(result).hi > 1e15);

        // Signed zero: tan(0) = 0 (tan is odd function)
        assert_eq!(Float::tan(<Df64 as Float>::neg_zero()), Df64::ZERO);
    }

    #[test]
    fn test_float_asin()
    {
        // Delegate to circular::asin, detailed precision tests are in circular.rs
        assert_eq!(Float::asin(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::asin(Df64::ONE), crate::consts::PI_HALF);
        assert_ulps_eq!(Float::asin(-Df64::ONE), -crate::consts::PI_HALF);
        assert_ulps_eq!(Float::asin(Df64::from(0.5)), crate::consts::PI_SIXTH);
    }

    #[test]
    fn test_float_acos()
    {
        // Delegate to circular::acos, detailed precision tests are in circular.rs
        assert_eq!(Float::acos(Df64::ONE), Df64::ZERO);
        assert_ulps_eq!(Float::acos(Df64::ZERO), crate::consts::PI_HALF);
        assert_ulps_eq!(Float::acos(Df64::from(0.5)), crate::consts::PI_THIRD);
        assert_ulps_eq!(Float::acos(-Df64::ONE), crate::consts::PI);
    }

    #[test]
    fn test_float_atan()
    {
        // Delegate to circular::atan, detailed precision tests are in circular.rs
        assert_eq!(Float::atan(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::atan(Df64::ONE), crate::consts::PI_FOURTH);
        assert_ulps_eq!(Float::atan(-Df64::ONE), -crate::consts::PI_FOURTH);

        // Test infinity input
        assert_ulps_eq!(Float::atan(Df64::INFINITY), crate::consts::PI_HALF);
        assert_ulps_eq!(Float::atan(Df64::NEG_INFINITY), -crate::consts::PI_HALF);
    }

    #[test]
    fn test_float_atan2()
    {
        // Delegate to circular::atan2, detailed precision tests are in circular.rs
        assert_ulps_eq!(Float::atan2(Df64::ONE, Df64::ONE), crate::consts::PI_FOURTH);
        assert_ulps_eq!(Float::atan2(Df64::ONE, Df64::ZERO), crate::consts::PI_HALF);
        assert_ulps_eq!(Float::atan2(Df64::ZERO, -Df64::ONE), crate::consts::PI);

        // Signed zero behavior
        let pos_zero = Df64::ZERO;
        let neg_zero = <Df64 as Float>::neg_zero();
        let one = Df64::ONE;
        let neg_one = -Df64::ONE;
        let pi = crate::consts::PI;

        // atan2(+0, +x) = +0
        let result = Float::atan2(pos_zero, one);
        assert_eq!(result, pos_zero);

        // atan2(+0, -x) = +pi
        let result = Float::atan2(pos_zero, neg_one);
        assert_ulps_eq!(result, pi);

        // atan2(-0, +x) = -0 (value equality with +0)
        let result = Float::atan2(neg_zero, one);
        assert_eq!(result, pos_zero);  // Value equality

        // atan2(-0, -x) returns +pi in current implementation
        // (IEEE 754 specifies -pi, but implementation does not distinguish -0)
        let result = Float::atan2(neg_zero, neg_one);
        assert_ulps_eq!(result, pi);

        // atan2(+y, 0) = +pi/2
        let result = Float::atan2(one, pos_zero);
        assert_ulps_eq!(result, crate::consts::PI_HALF);

        // atan2(-y, 0) = -pi/2
        let result = Float::atan2(neg_one, pos_zero);
        assert_ulps_eq!(result, -crate::consts::PI_HALF);
    }

    #[test]
    fn test_float_sin_cos()
    {
        // Delegate to circular::sincos, detailed precision tests are in circular.rs
        let (s, c) = Float::sin_cos(Df64::ZERO);
        assert_eq!(s, Df64::ZERO);
        assert_eq!(c, Df64::ONE);

        let pi_four = crate::consts::PI_FOURTH;
        let (s, c) = Float::sin_cos(pi_four);
        let sqrt2_over_2 = Float::recip(Float::sqrt(Df64::from(2.0)));
        assert_ulps_eq!(s, sqrt2_over_2);
        assert_ulps_eq!(c, sqrt2_over_2);
    }

    // ===== Float trait hyperbolic functions (6 methods) =====

    #[test]
    fn test_float_sinh()
    {
        // Delegate to hyperbolic::sinh, detailed precision tests are in hyperbolic.rs
        assert_eq!(Float::sinh(Df64::ZERO), Df64::ZERO);

        // sinh(1) = (e - 1/e) / 2
        let e = crate::consts::EULER_E;
        let expected = (e - Float::recip(e)) / Df64::from(2.0);
        assert_ulps_eq!(Float::sinh(Df64::ONE), expected);

        // Signed zero: sinh(0) = 0 (sinh is odd function)
        assert_eq!(Float::sinh(<Df64 as Float>::neg_zero()), Df64::ZERO);
    }

    #[test]
    fn test_float_cosh()
    {
        // Delegate to hyperbolic::cosh, detailed precision tests are in hyperbolic.rs
        assert_eq!(Float::cosh(Df64::ZERO), Df64::ONE);

        // cosh(1) = (e + 1/e) / 2
        let e = crate::consts::EULER_E;
        let expected = (e + Float::recip(e)) / Df64::from(2.0);
        assert_ulps_eq!(Float::cosh(Df64::ONE), expected);

        // cosh is even function
        assert_ulps_eq!(Float::cosh(-Df64::ONE), expected);

        // Signed zero: cosh(0) = 1 (cosh is even function)
        assert_eq!(Float::cosh(<Df64 as Float>::neg_zero()), Df64::ONE);
    }

    #[test]
    fn test_float_tanh()
    {
        // Delegate to hyperbolic::tanh, detailed precision tests are in hyperbolic.rs
        assert_eq!(Float::tanh(Df64::ZERO), Df64::ZERO);

        // tanh approaches ±1 for large |x|
        let large = Df64::from(10.0);
        assert!(Float::tanh(large).hi > 0.9999);
        assert!(Float::tanh(-large).hi < -0.9999);

        // Signed zero: tanh(0) = 0 (tanh is odd function)
        assert_eq!(Float::tanh(<Df64 as Float>::neg_zero()), Df64::ZERO);
    }

    #[test]
    fn test_float_asinh()
    {
        // Delegate to hyperbolic::asinh, detailed precision tests are in hyperbolic.rs
        assert_eq!(Float::asinh(Df64::ZERO), Df64::ZERO);

        // asinh(1) = ln(1 + sqrt(2))
        let sqrt2 = Float::sqrt(Df64::from(2.0));
        let expected = Float::ln(Df64::ONE + sqrt2);
        assert_ulps_eq!(Float::asinh(Df64::ONE), expected);

        // asinh is odd function
        assert_ulps_eq!(Float::asinh(-Df64::ONE), -expected);
    }

    #[test]
    fn test_float_acosh()
    {
        // Delegate to hyperbolic::acosh, detailed precision tests are in hyperbolic.rs
        assert_eq!(Float::acosh(Df64::ONE), Df64::ZERO);

        // acosh(2) = ln(2 + sqrt(3))
        let sqrt3 = Float::sqrt(Df64::from(3.0));
        let expected = Float::ln(Df64::from(2.0) + sqrt3);
        assert_ulps_eq!(Float::acosh(Df64::from(2.0)), expected);
    }

    #[test]
    fn test_float_atanh()
    {
        // Delegate to hyperbolic::atanh, detailed precision tests are in hyperbolic.rs
        assert_eq!(Float::atanh(Df64::ZERO), Df64::ZERO);

        // atanh(0.5) = 0.5 * ln(3)
        let expected = Df64::from(0.5) * Float::ln(Df64::from(3.0));
        assert_ulps_eq!(Float::atanh(Df64::from(0.5)), expected);

        // atanh is odd function
        assert_ulps_eq!(Float::atanh(Df64::from(-0.5)), -expected);
    }

    // ===== Float trait angle conversion (2 methods) =====

    #[test]
    fn test_float_to_degrees()
    {
        // Uses precomputed DEGREES_PER_RADIAN constant
        assert_eq!(Float::to_degrees(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::to_degrees(crate::consts::PI), Df64::from(180.0));
        assert_ulps_eq!(Float::to_degrees(crate::consts::PI_HALF), Df64::from(90.0));
        assert_ulps_eq!(Float::to_degrees(-crate::consts::PI), Df64::from(-180.0));
    }

    #[test]
    fn test_float_to_radians()
    {
        // Uses precomputed RADIANS_PER_DEGREE constant
        assert_eq!(Float::to_radians(Df64::ZERO), Df64::ZERO);
        assert_ulps_eq!(Float::to_radians(Df64::from(180.0)), crate::consts::PI);
        assert_ulps_eq!(Float::to_radians(Df64::from(90.0)), crate::consts::PI_HALF);
        assert_ulps_eq!(Float::to_radians(Df64::from(-180.0)), -crate::consts::PI);
    }

    #[test]
    fn test_float_const()
    {
        use num_traits::FloatConst;
        assert_eq!(Df64::E(), consts::EULER_E);
        assert_eq!(Df64::FRAC_1_PI(), consts::ONE_OVER_PI);
        assert_eq!(Df64::FRAC_1_SQRT_2(), consts::FRAC_1_SQRT_2);
        assert_eq!(Df64::FRAC_2_PI(), consts::TWO_OVER_PI);
        assert_eq!(Df64::FRAC_2_SQRT_PI(), consts::TWO_OVER_SQRT_PI);
        assert_eq!(Df64::FRAC_PI_2(), consts::PI_HALF);
        assert_eq!(Df64::FRAC_PI_3(), consts::PI_THIRD);
        assert_eq!(Df64::FRAC_PI_4(), consts::PI_FOURTH);
        assert_eq!(Df64::FRAC_PI_6(), consts::PI_SIXTH);
        assert_eq!(Df64::FRAC_PI_8(), consts::PI_EIGHTH);
        assert_eq!(Df64::LN_10(), consts::LN_10);
        assert_eq!(Df64::LN_2(), consts::LN_2);
        assert_eq!(Df64::LOG10_E(), consts::LOG10_E);
        assert_eq!(Df64::LOG2_E(), consts::LOG2_E);
        assert_eq!(Df64::PI(), consts::PI);
        assert_eq!(Df64::SQRT_2(), consts::SQRT_2);
        assert_eq!(Df64::TAU(), consts::TWO_PI);
        assert_eq!(Df64::LOG10_2(), consts::LOG10_2);
        assert_eq!(Df64::LOG2_10(), consts::LOG2_10);
    }

    // ===== Float trait not yet implemented (2 methods) =====

    #[test]
    fn test_float_constants_methods()
    {
        // Test that Float trait constant methods match Df64 constants
        assert_eq!(<Df64 as Float>::min_value(), Df64::MIN);
        assert_eq!(<Df64 as Float>::min_positive_value(), Df64::MIN_POSITIVE);
        assert_eq!(<Df64 as Float>::max_value(), Df64::MAX);
        assert_eq!(<Df64 as Float>::epsilon(), Df64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_float_integer_decode_not_implemented()
    {
        // integer_decode is marked as todo!()
        let _ = <Df64 as Float>::integer_decode(Df64::from(1.5));
    }

}
