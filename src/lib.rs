use core::f64;

#[allow(non_camel_case_types)]
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct d64 {
    hi: f64,
    lo: f64
}

/// Arithmetic with compensated errors.
///
/// This trait marks a type as a compensated arithmetic type. It maintains the
/// result of an arithmetic operation plus a compensate, which is the
/// difference of the operation inside `T` and the exact result (or at least
/// more accurate result.)
///
/// The following operations are required:
///
///  | (op)       | method                      | Also known as | Exact? |
///  |------------|-----------------------------|---------------|--------|
///  | `a + b`    | `::compensated_sum(a, b)`   | 2sum(a, b)    | yes    |
///  | `a - b`    | `::compensated_diff(a, b)`  | 2diff(a, b)   | yes    |
///  | `a * b`    | `::compensated_prod(a, b)`  | 2prod(a, b)   | yes    |
///  | `a / b`    | `::compensated_ratio(a, b)` |               | no     |
///  | `a.sqrt()` | `::compensated_sqrt(a, b)`  |               | no     |
///
/// For sum and difference, there are `unsafe` versions, which may be faster
/// because they may assume that the arguments are ordered by magnitude, i.e,
/// `a.abs() >= b.abs()`. This constraint can be slightly relaxed.[^1]
///
///  | (op)    | unsafe method                   | Also known as     | Exact? |
///  |---------|---------------------------------|-------------------|--------|
///  | `a + b` | `::compensated_fast_sum(a, b)`  | fast2sum(a, b)    | yes*   |
///  | `a - b` | `::compensated_fast_diff(a, b)` | fast2diff(a, b)   | yes*   |
///
/// **Warning**: Compensated arithmetic is not guaranteed to conform to IEEE
/// rules when it comes to infinities. One usually gets NaN in this case.
///
/// [^1]: J.-M. Muller and L. Rideau, ACM Trans. Math. Softw. 48, 1, 9 (2022).
pub trait CompensatedArithmetic<T> : From<T> + Into<T>
{
    /// type of the compensate.
    type Compensate;

    /// Zero accumulator
    const ZERO: Self;

    /// One accumulator
    const ONE: Self;

    /// Return the compensate (lo part)
    ///
    /// Return the compensate, i.e., the difference of the current value and
    /// its `T` approximation, `self.into<T>()`.
    fn compensate(self: &Self) -> Self::Compensate;

    /// Add `a` and `b` while compensating exactly for the error.
    ///
    /// Adds two values in extended precision, where the result can be
    /// represented exactly. On floating point numbers, this is known as
    /// "2sum" or compensated summation.
    fn compensated_sum(a: T, b: T) -> Self;

    /// Subtract `b` from `a` while compensating exactly for the error.
    ///
    /// Subtracts two values in extended precision, where the result can be
    /// represented exactly.
    fn compensated_diff(a: T, b: T) -> Self;

    /// Multiply `a` with `b` while compensating exactly for the error.
    ///
    /// Multiplies two values in extended precision, where the result can be
    /// represented exactly.
    fn compensated_prod(a: T, b: T) -> Self;

    /// Divides `a` by `b` while compensating approximately for the error.
    ///
    /// Divides two values in extended precision.
    fn compensated_ratio(a: T, b: T) -> Self;

    /// Compensated square root operation
    ///
    /// Takes the square root and maintains correction term.
    fn compensated_sqrt(a: T) -> Self;

    /// Add `large` and `small` exactly, assuming `large.abs() >= small.abs()`.
    ///
    /// Adds a small value `small` to a large value `large` in extended
    /// precision, where the result can be represented exactly if `a` has
    /// larger magnitude than `b`. (For double-double arithmetic, this
    /// condition can be slightly relaxed.). On floating point numbers, this is
    /// known as Kahan summation or "fast2sum".
    ///
    /// **Safety**: you must make sure that large is indeed the larger number.
    unsafe fn compensated_fast_sum(large: T, small: T) -> Self {
        return Self::compensated_sum(large, small);
    }

    /// Subtract `small` from `large` exactly, assuming `large.abs() >= small.abs()`.
    ///
    /// Subtracts a small value `small` from a large value `large` in extended
    /// precision, where the result can be represented exactly if `a` has
    /// larger magnitude than `b`. (For double-double arithmetic, this
    /// condition can be slightly relaxed.).
    ///
    /// **Safety**: you must make sure that large is indeed the larger number.
    unsafe fn compensated_fast_diff(large: T, small: T) -> Self {
        return Self::compensated_diff(large, small);
    }
}

pub trait AddFast<T = Self> {
    type Output;

    // XXX: should maybe be unsafe?
    fn add_fast(self, small: T) -> Self::Output;
}

pub trait SubFast<T = Self> {
    type Output;

    // XXX: should maybe be unsafe?
    fn sub_fast(self, small: T) -> Self::Output;
}

#[cfg(test)]
mod test_utils;
mod utils;

pub mod arith;
pub mod circular;
pub mod checks;
pub mod consts;
pub mod exp;
pub mod funcs;
pub mod gauss;
pub mod hyperbolic;
pub mod roots;
pub mod round;
mod traits;

// Convert to float
impl From<d64> for f64 {
    fn from(src: d64) -> f64 {
        src.hi
    }
}

// Convert from float
impl From<f64> for d64 {
    fn from(src: f64) -> d64 {
        d64 {hi: src, lo: 0.0}
    }
}
