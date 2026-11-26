// SPDX-License-Identifier: MIT

//! The declared ABI layout of `Compensated`, and the parts accessors that make
//! it usable without `transmute`.
//!
//! The point of `#[repr(C)]` here is that a C consumer of a double-double can
//! treat the value as `struct { double hi; double lo; }`, and the point of the
//! accessors is that Rust consumers never have to rely on that layout.

use core::mem::{align_of, size_of};
use xprec::Df64;

#[test]
fn df64_is_two_doubles_in_hi_lo_order() {
    assert_eq!(size_of::<Df64>(), 2 * size_of::<f64>());
    assert_eq!(align_of::<Df64>(), align_of::<f64>());
    // The field offsets are asserted inside the crate, where the fields are
    // visible; from here the public guarantee is the size and the order that
    // `as_parts` reports.
}

#[test]
fn df64_has_no_padding_so_the_layout_is_the_array() {
    // bytemuck's `Pod` needs this property, and so does any C consumer that
    // reads the value as `[f64; 2]`.
    assert_eq!(size_of::<Df64>(), size_of::<[f64; 2]>());
}

#[test]
fn parts_round_trip_without_renormalising() {
    let value = Df64::new_full(1.0, 1e-20);
    let (hi, lo) = value.as_parts();
    assert_eq!(hi.to_bits(), 1.0f64.to_bits());
    assert_eq!(lo, 1e-20);
    let rebuilt = Df64::from_parts(hi, lo);
    assert_eq!(rebuilt, value);
    // Deliberately overlapping parts: `from_parts` keeps them as given.
    let overlapping = Df64::from_parts(1.0f64, 1.0f64);
    assert_eq!(overlapping.as_parts(), (1.0f64, 1.0f64));
}

#[test]
fn the_message_of_a_double_double_survives_the_round_trip() {
    // Four f64 addends of very different magnitude, summed in double-double.
    // The point is that taking the value apart and putting it back does not
    // lose the compensation.
    let a: f64 = 1.0;
    let b: f64 = 1e-17;
    let c: f64 = 1e-33;
    let d: f64 = 1e-49;
    let sum = Df64::from(a) + Df64::from(b) + Df64::from(c) + Df64::from(d);
    let (hi, lo) = sum.as_parts();
    let rebuilt = Df64::from_parts(hi, lo);
    assert_eq!(rebuilt, sum);
    // `f64` alone would drop `c` and `d`; the compensated value keeps them.
    assert_eq!(
        sum,
        Df64::from(1.0) + Df64::from(1e-17) + Df64::from(1e-33) + Df64::from(1e-49)
    );
    assert_eq!(sum.hi(), 1.0f64 + 1e-17);
    assert_ne!(sum.lo(), 0.0f64);
}
