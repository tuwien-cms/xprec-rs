//! Integration test for the built-in `From`/`TryFrom` conversions of `Df64`
//! (issue #20).
//!
//! Like `inherent_api.rs`, this file must not import `num_traits`: the point
//! is that the conversions work through the standard library traits alone.
//!
//! Copyright (C) 2023-2025 Markus Wallerberger and others
//! SPDX-License-Identifier: MIT

use xprec::Df64;

#[test]
fn from_every_integer_type() {
    assert_eq!(Df64::from(42i8), Df64::from(42.0));
    assert_eq!(Df64::from(42i16), Df64::from(42.0));
    assert_eq!(Df64::from(42i32), Df64::from(42.0));
    assert_eq!(Df64::from(42i64), Df64::from(42.0));
    assert_eq!(Df64::from(42isize), Df64::from(42.0));
    assert_eq!(Df64::from(-42i32), Df64::from(-42.0));

    assert_eq!(Df64::from(42u8), Df64::from(42.0));
    assert_eq!(Df64::from(42u16), Df64::from(42.0));
    assert_eq!(Df64::from(42u32), Df64::from(42.0));
    assert_eq!(Df64::from(42u64), Df64::from(42.0));
    assert_eq!(Df64::from(42usize), Df64::from(42.0));
}

#[test]
fn from_i64_keeps_the_low_bits() {
    // 2^53 + 1 is not representable in f64, so the compensation is required to
    // survive the round trip.
    let value = (1i64 << 53) + 1;
    let x = Df64::from(value);
    assert_eq!(x.hi(), (1i64 << 53) as f64);
    assert_eq!(x.lo(), 1.0);
    assert_eq!(i64::try_from(x).unwrap(), value);
    assert!(x > Df64::from((1i64 << 53) as f64));

    let big = (1u64 << 62) + 1;
    let y = Df64::from(big);
    assert_eq!(u64::try_from(y).unwrap(), big);
}

#[test]
fn try_from_round_trips() {
    for value in [0i64, 1, -1, 12345, -12345, i32::MAX as i64, i32::MIN as i64] {
        assert_eq!(i64::try_from(Df64::from(value)).unwrap(), value);
    }
    for value in [0u64, 1, 12345, u32::MAX as u64, 1u64 << 63] {
        assert_eq!(u64::try_from(Df64::from(value)).unwrap(), value);
    }

    assert_eq!(i8::try_from(Df64::from(-7i8)).unwrap(), -7);
    assert_eq!(u8::try_from(Df64::from(255u8)).unwrap(), 255);
    assert_eq!(i16::try_from(Df64::from(1000i16)).unwrap(), 1000);
    assert_eq!(u16::try_from(Df64::from(1000u16)).unwrap(), 1000);
    assert_eq!(i32::try_from(Df64::from(1_000_000i32)).unwrap(), 1_000_000);
    assert_eq!(u32::try_from(Df64::from(1_000_000u32)).unwrap(), 1_000_000);
    assert_eq!(isize::try_from(Df64::from(-99isize)).unwrap(), -99);
    assert_eq!(usize::try_from(Df64::from(99usize)).unwrap(), 99);
    assert_eq!(i128::try_from(Df64::from(123_456_789i64)).unwrap(), 123_456_789);
    assert_eq!(u128::try_from(Df64::from(123_456_789u64)).unwrap(), 123_456_789);
}

#[test]
fn try_from_discards_the_fractional_part() {
    assert_eq!(i64::try_from(Df64::from(2.75)).unwrap(), 2);
    assert_eq!(i64::try_from(Df64::from(-2.75)).unwrap(), -2);
}

#[test]
fn try_from_rejects_out_of_range_values() {
    assert!(i8::try_from(Df64::from(200.0)).is_err());
    assert!(i8::try_from(Df64::from(-200.0)).is_err());
    assert!(u8::try_from(Df64::from(-1.0)).is_err());
    assert!(u64::try_from(Df64::from(-1.0)).is_err());
    assert!(i64::try_from(Df64::INFINITY).is_err());
    assert!(i64::try_from(Df64::NEG_INFINITY).is_err());
    assert!(i64::try_from(Df64::NAN).is_err());
}

#[test]
fn error_reports_the_value() {
    let value = Df64::from(200.0);
    let error = i8::try_from(value).unwrap_err();
    assert_eq!(error.value, value);
    assert!(format!("{error}").contains("200"));
}

#[test]
fn into_f64_still_works() {
    let x = Df64::from(2.5);
    let y: f64 = x.into();
    assert_eq!(y, 2.5);

    // ... and `From<f64>` is not ambiguous with the integer conversions
    let z: Df64 = 2.5f64.into();
    assert_eq!(z, x);
}
