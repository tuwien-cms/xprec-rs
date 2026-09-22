use super::Df64;

impl Df64 {
    /// Machine epsilon ϵ for Df64.
    ///
    /// Lowest positive number `ϵ` such that for any normal number `x`,
    /// `(1 + ϵ) * x` is guaranteed to be distinct from `x`. Note that compared
    /// to IEEE floats (`f32`, `f64`), there are two complications:
    ///
    ///  1. arithmetic operations on Df64 have an error bound of up to 3ϵ
    ///     rather than ϵ/2 in the case of regular floats.
    ///
    ///  2. `(1 + ϵ) * x` is not necessarily the next distinct Df64 value.
    ///
    pub const EPSILON: Df64 = Df64 {
        hi: f64::EPSILON * f64::EPSILON / 2.0, lo: 0.0
    };

    /// Infinity (∞)
    pub const INFINITY: Df64 = Df64 {
        hi: f64::INFINITY, lo: 0.0
    };

    /// Negative infinity (∞)
    pub const NEG_INFINITY: Df64 = Df64 {
        hi: f64::NEG_INFINITY, lo: 0.0
    };

    /// Not a Number (NaN)
    ///
    /// Note that this is only one of the possible NaN values, and a quiet one.
    pub const NAN: Df64 = Df64 {
        hi: f64::NAN, lo: 0.0
    };

    /// Largest finite value.
    pub const MAX: Df64 = Df64 {
        hi: f64::MAX, lo: f64::MAX * f64::EPSILON / 4.0
    };

    /// Largest negative value by magnitude.
    pub const MIN: Df64 = Df64 {
        hi: f64::MIN, lo: f64::MIN * f64::EPSILON / 4.0
    };

    /// Smallest positive normal value.
    ///
    /// Note that Df64 has a smaller range of normal numbers than f64, because
    /// it implies that both hi and lo part must be normal.
    pub const MIN_POSITIVE: Df64 = Df64 {
        hi: f64::MIN_POSITIVE / f64::EPSILON, lo: 0.0
    };

    /// The radix or base of the internal representation.
    pub const RADIX: u32 = f64::RADIX;

    /// Maximum exponent.
    ///
    /// Largest exponent `e` such that for any `m.abs() < 1`, `ldexp(m, e)` is
    /// a normal number, i.e., does not overflow.
    pub const MAX_EXP: i32 = f64::MAX_EXP;

    /// Minimum exponent.
    ///
    /// Smallest exponent `e` such that for any `m.abs() < 1`, `ldexp(m, e)` is
    /// a normal number, i.e., does not underflow or go into the subnormals.
    pub const MIN_EXP: i32 = f64::MIN_EXP + f64::MANTISSA_DIGITS as i32 - 2;

    /// Zero
    pub const ZERO: Df64 = Df64 {hi: 0.0, lo: 0.0};

    /// One
    pub const ONE: Df64 = Df64 {hi: 1.0, lo: 0.0};

}

/// Circle number π
pub const PI: Df64 = Df64 {hi: 3.141592653589793, lo: 1.2246467991473532e-16};

/// Two times π
pub const TWO_PI: Df64 = Df64 {hi: 6.283185307179586, lo: 2.4492935982947064e-16};

/// Half of π
pub const PI_HALF: Df64 = Df64 {hi: 1.5707963267948966, lo: 6.123233995736766e-17};

/// One third of π
pub const PI_THIRD: Df64 = Df64 {hi: 1.0471975511965979, lo: -1.072081766451091e-16};

/// One fourth of π
pub const PI_FOURTH: Df64 = Df64 {hi: 0.7853981633974483, lo: 3.061616997868383e-17};

/// One sixth of π
pub const PI_SIXTH: Df64 = Df64 {hi: 0.5235987755982989, lo: -5.360408832255455e-17};

/// One eighth of π
pub const PI_EIGHTH: Df64 = Df64 {hi: 0.39269908169872414, lo: 1.5308084989341915e-17};

/// Reciprocal of π
pub const ONE_OVER_PI: Df64 = Df64 {hi: 0.3183098861837907, lo: -1.9678676675182486e-17};

/// Twice the reciprocal of π
pub const TWO_OVER_PI: Df64 = Df64 {hi: 0.6366197723675814, lo: -3.935735335036497e-17};

/// Twice the reciprocal of the square root of π
pub const TWO_OVER_SQRT_PI: Df64 = Df64 {hi: 1.1283791670955126, lo: 1.533545961316588e-17};

/// Euler number e
pub const EULER_E: Df64 = Df64 {hi: 2.718281828459045, lo: 1.4456468917292502e-16};

/// Binary logarithm of e
pub const LOG2_E: Df64 = Df64 {hi: 1.4426950408889634, lo: 2.0355273740931033e-17};

/// Logarithm base-10 of e
pub const LOG10_E: Df64 = Df64 {hi: 0.4342944819032518, lo: 1.098319650216765e-17};

/// Natural logarithm of 2
pub const LN_2: Df64 = Df64 {hi: 0.6931471805599453, lo: 2.3190468138462996e-17};

/// Natural logarithm of 10
pub const LN_10: Df64 = Df64 {hi: 2.302585092994046, lo: -2.1707562233822494e-16};

/// Logarithm base-10 of 2
pub const LOG10_2: Df64 = Df64 {hi: 0.30102999566398120, lo: -2.8037281277851704e-18};

/// Logarithm base-2 of 10
pub const LOG2_10: Df64 = Df64 {hi: 3.321928094887362, lo: 1.661617516973592e-16};

/// Square root of 2
pub const SQRT_2: Df64 = Df64 {hi: 1.4142135623730951, lo: -9.667293313452913e-17};

/// Reciprocal of square root of 2 (1/√2 = √2/2)
pub const FRAC_1_SQRT_2: Df64 = Df64 {hi: 0.7071067811865476, lo: -4.833646656726457e-17};

/// Radians per degree (π/180)
pub const RADIANS_PER_DEGREE: Df64 = Df64 {hi: 0.017453292519943295, lo: 2.9486522708701687e-19};

/// Degrees per radian (180/π)
pub const DEGREES_PER_RADIAN: Df64 = Df64 {hi: 57.29577951308232, lo: -1.9878495670576283e-15};


#[cfg(test)]
mod test
{
    use super::*;
    use crate::*;
    use approx::*;

    #[test]
    fn test_consts()
    {
        assert_ulps_eq!(circular::sin(PI_HALF), Df64::ONE);
        assert_ulps_eq!(circular::cos(PI), -Df64::ONE);
        assert_ulps_eq!(2.0 * PI, TWO_PI);
        assert_ulps_eq!(2.0 * PI_HALF, PI);
        assert_ulps_eq!(3.0 * PI_THIRD, PI);
        assert_ulps_eq!(4.0 * PI_FOURTH, PI);
        assert_ulps_eq!(6.0 * PI_SIXTH, PI);
        assert_ulps_eq!(8.0 * PI_EIGHTH, PI);
        assert_ulps_eq!(ONE_OVER_PI, arith::reciprocal_q(PI));
        assert_ulps_eq!(TWO_OVER_PI, arith::reciprocal_q(PI_HALF));
        assert_ulps_eq!(TWO_OVER_SQRT_PI, roots::inv_sqrt(PI_FOURTH));
        assert_ulps_eq!(LOG2_E, arith::reciprocal_q(LN_2));
        assert_ulps_eq!(LOG10_E, arith::reciprocal_q(LN_10));
        assert_ulps_eq!(exp::exp(Df64::ONE), EULER_E);
        assert_ulps_eq!(RADIANS_PER_DEGREE, arith::div_qd(PI, 180.0));
        assert_ulps_eq!(DEGREES_PER_RADIAN, arith::mul_qd(ONE_OVER_PI, 180.0));
        assert_ulps_eq!(SQRT_2, arith::sqrt_q(Df64::from(2.0)));
        assert_ulps_eq!(FRAC_1_SQRT_2, roots::inv_sqrt(Df64::from(2.0)));
        assert_ulps_eq!(LOG10_2, LOG10_E * LN_2);
        assert_ulps_eq!(LOG2_10, LOG2_E * LN_10);
    }
}
