//! Integration test for the inherent methods of `Df64` (issue #20).
//!
//! **This file must not import `num_traits` or `simba`.**  The whole point of
//! the inherent methods is that they can be called with nothing but
//! `use xprec::Df64;`; if any of them were only available through a trait,
//! this file would fail to compile.
//!
//! Copyright (C) 2023-2025 Markus Wallerberger and others
//! SPDX-License-Identifier: MIT

use xprec::Df64;

/// Relative check that does not need an external crate.  `ulps` counts
/// multiples of `Df64::EPSILON`.
fn assert_close(actual: Df64, expected: f64, ulps: f64) {
    let value = actual.hi() + actual.lo();
    let tol = ulps * 1.3e-32;
    if expected == 0.0 {
        assert!(
            value.abs() <= tol,
            "expected 0, got {actual} (absolute deviation {:.3e})",
            value.abs()
        );
    } else {
        let rel = ((value - expected) / expected.abs()).abs();
        assert!(
            rel <= tol,
            "expected {expected}, got {actual} (relative deviation {rel:.3e})"
        );
    }
}

#[test]
fn classification() {
    assert!(Df64::NAN.is_nan());
    assert!(!Df64::ONE.is_nan());

    assert!(Df64::INFINITY.is_infinite());
    assert!(Df64::NEG_INFINITY.is_infinite());
    assert!(!Df64::ONE.is_infinite());

    assert!(Df64::ONE.is_finite());
    assert!(!Df64::NAN.is_finite());

    assert!(Df64::ONE.is_normal());
    assert!(!Df64::ZERO.is_normal());

    assert!(!Df64::ONE.is_subnormal());
    assert!(!Df64::ZERO.is_subnormal());

    assert!(matches!(Df64::NAN.classify(), std::num::FpCategory::Nan));
    assert!(matches!(Df64::ZERO.classify(), std::num::FpCategory::Zero));
    assert!(matches!(
        Df64::INFINITY.classify(),
        std::num::FpCategory::Infinite
    ));

    assert!(Df64::ONE.is_sign_positive());
    assert!(!Df64::ONE.is_sign_negative());
    assert!((-Df64::ONE).is_sign_negative());
}

#[test]
fn basic_arithmetic() {
    assert_eq!(Df64::from(-3.0).abs(), Df64::from(3.0));
    assert_eq!(Df64::from(3.0).signum(), Df64::ONE);
    assert_eq!(Df64::from(-3.0).signum(), -Df64::ONE);
    assert_eq!(Df64::from(3.0).copysign(-Df64::ONE), Df64::from(-3.0));
    assert_eq!(Df64::from(4.0).recip(), Df64::from(0.25));
    assert_eq!(Df64::from(4.0).sqrt(), Df64::from(2.0));

    assert_eq!(
        Df64::from(3.0).mul_add(Df64::from(4.0), Df64::from(-5.0)),
        Df64::from(7.0)
    );

    assert_close(Df64::from(3.0).hypot(Df64::from(4.0)), 5.0, 4.0);
    assert_eq!(Df64::from(2.0).abs_sub(Df64::from(5.0)), Df64::ZERO);
    assert_eq!(Df64::from(5.0).abs_sub(Df64::from(2.0)), Df64::from(3.0));
}

#[test]
fn rounding() {
    let x = Df64::from(2.5);
    assert_eq!(x.floor(), Df64::from(2.0));
    assert_eq!(x.ceil(), Df64::from(3.0));
    assert_eq!(x.round(), Df64::from(3.0));
    assert_eq!((-x).trunc(), Df64::from(-2.0));
    assert_eq!(x.fract(), Df64::from(0.5));
    assert_eq!(Df64::from(-2.5).round(), Df64::from(-3.0));
}

#[test]
fn comparison() {
    assert_eq!(Df64::from(1.0).min(Df64::from(2.0)), Df64::ONE);
    assert_eq!(Df64::from(1.0).max(Df64::from(2.0)), Df64::from(2.0));
    assert_eq!(
        Df64::from(5.0).clamp(Df64::ZERO, Df64::from(3.0)),
        Df64::from(3.0)
    );
}

#[test]
fn exponential_and_logarithmic() {
    assert_close(Df64::ONE.exp(), std::f64::consts::E, 1.0);
    assert_eq!(Df64::ZERO.exp(), Df64::ONE);
    assert_close(Df64::ONE.exp2(), 2.0, 4.0);
    assert_eq!(Df64::ZERO.exp_m1(), Df64::ZERO);
    assert_eq!(Df64::ONE.ln(), Df64::ZERO);
    assert_eq!(Df64::ZERO.ln_1p(), Df64::ZERO);
    assert_eq!(Df64::ONE.log(Df64::from(7.0)), Df64::ZERO);
    assert_close(Df64::from(8.0).log2(), 3.0, 4.0);
    assert_close(Df64::from(1000.0).log10(), 3.0, 4.0);

    assert_close(Df64::from(2.0).powi(10), 1024.0, 8.0);
    assert_close(Df64::from(2.0).powi(-1), 0.5, 1.0);
    assert_close(Df64::from(2.0).powf(Df64::from(10.0)), 1024.0, 16.0);
    assert_close(Df64::from(9.0).powf(Df64::from(0.5)), 3.0, 8.0);
}

#[test]
fn trigonometric_and_hyperbolic() {
    let quarter = Df64::from(std::f64::consts::FRAC_PI_2);
    assert_close(quarter.sin(), 1.0, 8.0);
    // Note that `FRAC_PI_2` differs from pi/2 by ~6e-17, so the cosine is
    // small but not zero.
    assert!(quarter.cos().abs().hi() < 1e-16);
    assert_close(Df64::from(1.0).tan().atan(), 1.0, 8.0);

    let (sin, cos) = quarter.sin_cos();
    assert_close(sin, 1.0, 8.0);
    assert!(cos.abs().hi() < 1e-16);

    assert_close(Df64::from(1.0).atan2(Df64::ONE), std::f64::consts::FRAC_PI_4, 8.0);

    assert_eq!(Df64::ZERO.sinh(), Df64::ZERO);
    assert_close(Df64::ZERO.cosh(), 1.0, 1.0);
    assert_eq!(Df64::ZERO.tanh(), Df64::ZERO);
    assert_eq!(Df64::ZERO.asinh(), Df64::ZERO);
    assert_eq!(Df64::ONE.acosh(), Df64::ZERO);
    assert_eq!(Df64::ZERO.atanh(), Df64::ZERO);
    assert_close(Df64::from(1.0).sinh().asinh(), 1.0, 8.0);
    assert_close(Df64::from(2.0).cosh().acosh(), 2.0, 8.0);
    assert_close(Df64::from(0.5).tanh().atanh(), 0.5, 8.0);
}

#[test]
fn angle_conversion() {
    let pi = Df64::from(std::f64::consts::PI);
    assert_close(pi.to_degrees(), 180.0, 4.0);
    assert_close(Df64::from(180.0).to_radians(), std::f64::consts::PI, 4.0);
}

#[test]
fn works_with_from_and_into() {
    // `From`/`Into` are built-in traits, so these need no import either.
    let x: Df64 = Df64::from(3.0);
    let y: f64 = x.into();
    assert_eq!(y, 3.0);
}
