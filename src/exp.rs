use super::Df64;
use super::arith::{
    addfast_dq, addfast_qd, addfast_qq, mul_pow2, reciprocal_q, square_q, subfast_qd,
    subfast_qq,
};
use super::checks::is_nan;
use super::consts;
use super::funcs;
use super::utils::reciprocal_factorial;
use libm::ldexp;

// The value of MAX.ln().
pub const LOG_MAX: f64 = 709.782712893384;

/// Exponential function `exp(x)`
pub fn exp(x: Df64) -> Df64
{
    // Now we perform checks for special values. Using not <= instead of >
    // also catches NaNs.
    if !(x.hi.abs() <= LOG_MAX) {
        if is_nan(x) {
            return x;
        } else if x.hi > 0.0 {
            return Df64::INFINITY;
        } else {
            return Df64::ZERO;
        }
    }

    let (m, expm1_y) = exp_split(x);
    let exp_m = ldexp(1.0, m);
    let exp_y = addfast_dq(1.0, expm1_y);
    let exp_x = if exp_m.is_finite() {
        mul_pow2(exp_y, exp_m)
    } else {
        funcs::ldexp(exp_y, m)
    };
    return exp_x;
}

/// Shifted exponential function `exp(x) - 1` without intermediate rounding
pub fn expm1(x: Df64) -> Df64
{
    // Now we perform checks for special values. Using not <= instead of >
    // also catches NaNs.
    if !(x.hi.abs() <= LOG_MAX) {
        if is_nan(x) {
            return x;
        } else if x.hi > 0.0 {
            return Df64::INFINITY;
        } else {
            return Df64::from(-1.0);
        }
    }

    let (m, expm1_y) = exp_split(x);

    // If m == 0, then it means we can and should use the expm1 kernel
    // directly, otherwise it is okay to simply subtract 1.0
    if m == 0 {
        return expm1_y;
    } else {
        let exp_m = ldexp(1.0, m);
        let exp_y = addfast_dq(1.0, expm1_y);
        let exp_x = if exp_m.is_finite() {
            mul_pow2(exp_y, exp_m)
        } else {
            funcs::ldexp(exp_y, m)
        };

        // XXX dispatch based on magnitude
        return exp_x - 1.0;
    }
}

#[inline]
pub fn exp2(x: Df64) -> Df64
{
    // XXX this is not great, since it loses precision for large values
    return exp(consts::LN_2 * x);
}

/// Returns true if `x` is exactly an integer value.
#[inline]
fn is_integer(x: Df64) -> bool
{
    // The compensation has to vanish as well: `1 + 1e-40` is not an integer
    // even though its high part is one.  `fract()` is NaN for infinities, so
    // this also rejects them.
    return x.hi.fract() == 0.0 && x.lo == 0.0;
}

/// Returns true if `x` is exactly an odd integer value.
#[inline]
fn is_odd_integer(x: Df64) -> bool
{
    // For |x| >= 2^53 every representable value is an even integer, which is
    // what the remainder evaluates to.
    return is_integer(x) && (x.hi % 2.0) != 0.0;
}

/// Power `base^n` with an arbitrary real exponent.
///
/// Special values and signs follow IEEE-754 / `f64::powf`.  The remaining
/// cases are computed as `exp(expo * log(base))`; that expression is not
/// defined for a negative base, so the sign is factored out first.
pub fn powf(base: Df64, expo: Df64) -> Df64
{
    // x^0 == 1 and 1^y == 1, also for NaN arguments.
    if expo.hi == 0.0 {
        return Df64::ONE;
    }
    if base == Df64::ONE {
        return Df64::ONE;
    }
    if base.hi.is_nan() || expo.hi.is_nan() {
        return Df64::NAN;
    }

    if base.hi == 0.0 {
        // (+-0)^y is +-0 or +-infinity, with the sign negative only for -0
        // raised to an odd integer.
        let neg = base.hi.is_sign_negative() && is_odd_integer(expo);
        return if expo.hi > 0.0 {
            if neg { Df64::from(-0.0) } else { Df64::ZERO }
        } else if neg {
            Df64::NEG_INFINITY
        } else {
            Df64::INFINITY
        };
    }

    if expo.hi.is_infinite() {
        // |x|^+-infinity is 0 or infinity, with the exact value 1 in between.
        let abs_base = funcs::abs(base);
        if abs_base == Df64::ONE {
            return Df64::ONE;
        }
        let underflow = (abs_base < Df64::ONE) == (expo.hi > 0.0);
        return if underflow { Df64::ZERO } else { Df64::INFINITY };
    }

    if base.hi.is_infinite() {
        // (+-infinity)^y, again with the sign negative only for an odd
        // integer exponent.
        let neg = base.hi.is_sign_negative() && is_odd_integer(expo);
        return if expo.hi > 0.0 {
            if neg { Df64::NEG_INFINITY } else { Df64::INFINITY }
        } else if neg {
            Df64::from(-0.0)
        } else {
            Df64::ZERO
        };
    }

    if base.hi < 0.0 {
        if !is_integer(expo) {
            // A negative base with a non-integer exponent has no real result.
            return Df64::NAN;
        }
        // Negate rather than multiply by +-1: `1.0 * Df64::INFINITY` is NaN
        // in compensated arithmetic, which would turn an overflow into NaN.
        let result = exp(expo * log(-base));
        return if is_odd_integer(expo) { -result } else { result };
    }

    return exp(expo * log(base));
}

/// Power `base^n` with an integer exponent.
#[inline]
pub fn powi(base: Df64, expo: i32) -> Df64
{
    if expo == 0 {
        return Df64::ONE;
    }
    if base.hi.is_nan() {
        return Df64::NAN;
    }

    let odd = expo % 2 != 0;
    if base.hi == 0.0 {
        let neg = base.hi.is_sign_negative() && odd;
        return if expo > 0 {
            if neg { Df64::from(-0.0) } else { Df64::ZERO }
        } else if neg {
            Df64::NEG_INFINITY
        } else {
            Df64::INFINITY
        };
    }
    if base.hi.is_infinite() {
        let neg = base.hi.is_sign_negative() && odd;
        return if expo > 0 {
            if neg { Df64::NEG_INFINITY } else { Df64::INFINITY }
        } else if neg {
            Df64::from(-0.0)
        } else {
            Df64::ZERO
        };
    }

    // Cheap exact cases: the general path goes through the logarithm, which
    // loses precision even where the result is representable exactly.  They
    // have to come after the zero and infinity branches above, because
    // `1/0` and `1/inf` are NaN in compensated arithmetic.
    if expo == 1 {
        return base;
    }
    if expo == -1 {
        return reciprocal_q(base);
    }

    // Don't use squaring - terrible roundoff properties.  The sign of a
    // negative base is carried by the exponent.  Negate rather than multiply
    // by +-1: `1.0 * Df64::INFINITY` is NaN in compensated arithmetic, which
    // would turn an overflow into NaN (e.g. `powi(2, 1024)`).
    let result = exp(expo as f64 * log(funcs::abs(base)));
    return if base.hi < 0.0 && odd { -result } else { result };
}

/// Returns significand and exponent of `exp`.
///
/// Given some argument `x`, returns a tuple `(m, y)`, such that the value of
/// the exponential function is given by:
///
///   exp(x) == pow(2, m) * (1.0 + y),
///
/// where the significant `-0.5 < y < 0.5` is accurate to full relative
/// precision and `m` is an integer which can be larger than the f64 range
/// but must be smaller than 2<<24.
pub(crate) fn exp_split(x: Df64) -> (i32, Df64)
{
    // Here is the main strategy. Let α be log(2)/128. Then we first reduce the
    // argument x modulo α, i.e.:
    //
    //     x = k * α + y
    //
    let (k, y) = reduce_mod_alpha(x);

    // We further split k = 128 * m + n, where `n` is between {0, ..., 127}
    // Then we have that:
    //
    //     exp(x) = ldexp(1, m) * exp(n * ALPHA + y)
    //
    let (m, n) = reduce_mod_128(k as i32);

    // Now compute expm1 for the small argument to full precision
    let expm1_y = expm1_small(n, y);
    return (m, expm1_y);
}

/// Natural logarithm `log(x)`
pub fn log(x: Df64) -> Df64
{
    // Start with logarithm of hi part
    let log_x0 = x.hi.ln();
    if !log_x0.is_finite() {
        return Df64::from(log_x0);
    }

    // Abramowitz and Stegun give the following series expansion (4.1.30):
    //
    //   log(x) = log(x0) + 2 (x - x0)/(x + x0) + O(x - x0)^3
    //
    let x0 = exp(Df64::from(log_x0));
    let corr = mul_pow2(subfast_qq(x, x0) / addfast_qq(x, x0), 2.0);
    let log_x = log_x0 + corr;
    return log_x;
}

/// Logarithm base-2
#[inline]
pub fn log2(x: Df64) -> Df64
{
    // Loses a little precision, but anyway seldom used
    return log(x) * consts::LOG2_E;
}

/// Logarithm base-10
#[inline]
pub fn log10(x: Df64) -> Df64
{
    // Loses a little precision, but anyway seldom used
    return log(x) * consts::LOG10_E;
}

/// Logarithm custom base
pub fn log_base(x: Df64, base: Df64) -> Df64
{
    // Loses precision, but seldom used.
    return log(x) / log(base);
}

/// Natural log of shifted argument `log(x + 1)` without intermediate rounding.
pub fn log1p(x: Df64) -> Df64
{
    // Start with logarithm of hi part
    let log_x0 = x.hi.ln_1p();
    if !log_x0.is_finite() {
        return Df64::from(log_x0);
    }

    // Again, we can use the same correction, but log1p <-> expm1
    //
    //   log(1 + x) = log(1 + x0) + 2 (x - x0)/(2 + x + x0) + O(x - x0)^3
    //
    // One need not worry about cancellation in the denominator for
    // x close to -1, since that is where we have an intrinsic loss of
    // precision anyway
    let x0 = expm1(Df64::from(log_x0));
    let corr = mul_pow2(subfast_qq(x, x0) / addfast_qq(2.0 + x, x0), 2.0);
    let log_x = log_x0 + corr;
    return log_x;
}

fn reduce_mod_128(k: i32) -> (i32, i32)
{
    let mut m = k >> 7;
    let mut n = k & 0x7F;
    if k & 0x40 != 0 {
        n -= 0x80;
        m += 1;
    }
    return (m, n);
}

fn reduce_mod_alpha(x: Df64) -> (f64, Df64)
{
    // ALPHA_T is an approximation of log(2)/128 to 90 significant bits -- 17
    // bits fewer than full double-double precision.  Observe then that
    // 128*log(DBL_MAX) is around 91000, which fit comfortably into 17 bits.
    // That means that the reduction of x modulo ALPHA_T:
    //
    //     x = n * ALPHA_T + z
    //
    // is *exact* for any x in the range of the exponential funcion.  We have
    // to correct this expression to at least 124 digits. The correction term
    // only needs to be in double precision
    //
    //     z = n * ALPHA_CORR + y
    //
    const INV_ALPHA: f64 = 184.6649652337873;
    const ALPHA_T: Df64 = Df64 {hi: 0.0054152123481245725, lo: 1.8117553232937405e-19};
    const ALPHA_CORR: f64 = 2.3681038446414578e-30;

    // maybe use ceil here instead
    let n = (x.hi * INV_ALPHA).round();
    let z_raw = subfast_qq(x, n * ALPHA_T);
    let z = subfast_qd(z_raw, n * ALPHA_CORR);
    return (n, z);
}

fn expm1_small(n: i32, y: Df64) -> Df64
{
    // Assuming a reduction mod α = log(2)/128:
    //
    //     x = n * α + y,
    //
    // the idea is to use the identity
    //
    //     expm1(x) = expm1(n * α) + exp(n * α) * expm1(y)
    //
    // to reduce the expansion order.
    assert!(2.0 * y.hi.abs() <= 0.0054152123481245725);
    let expm1_n = expm1_alphas(n);
    let exp_n = addfast_dq(1.0, expm1_n);
    let expm1_y = expm1_kernel(y, 6, 10);
    return addfast_qq(expm1_n, expm1_y * exp_n);
}

fn expm1_kernel(x: Df64, nquad: i32, n: i32) -> Df64
{
    assert!(x.hi.abs() < 1.0);

    // r = x
    let mut r = x;

    // r += x * x / 2
    let mut xpow = square_q(x);
    r = addfast_qq(r, mul_pow2(xpow, 0.5));

    // r += x^k / k!
    for k in 3..nquad+1 {
        xpow = xpow * x;
        r = addfast_qq(r, reciprocal_factorial(k) * xpow);
    }

    // Here the terms are so small that they only affect the lo part, so
    // we can get away with double arithmetic.
    let mut r_d: f64 = 0.0;
    let mut xpow_d = xpow.hi;
    for k in nquad+1..n+1 {
        xpow_d *= x.hi;
        r_d += reciprocal_factorial(k).hi * xpow_d;
    }
    r = addfast_qd(r, r_d);
    return r;
}

#[inline]
const fn expm1_alphas(n: i32) -> Df64
{
    // For multiples of alpha = log(2)/128, precompute and store the
    // exponential function in a table, from -64*alpha until 64*alpha
    const EXPM1_ALPHAS : [Df64; 128] = [
        Df64 { hi: -0.2928932188134525, lo: 7.174684663993261e-18 },
        Df64 { hi: -0.2890536989154172, lo: -8.038914457945122e-18 },
        Df64 { hi: -0.285193330804015, lo: -6.0158212445268276e-18 },
        Df64 { hi: -0.2813120012755088, lo: -2.1020170082337783e-17 },
        Df64 { hi: -0.2774095965114767, lo: -1.5118790674969937e-17 },
        Df64 { hi: -0.27348600207547374, lo: 2.66114081842773e-17 },
        Df64 { hi: -0.26954110290967653, lo: 2.7509265300881745e-17 },
        Df64 { hi: -0.265574783331509, lo: -1.318173744858969e-17 },
        Df64 { hi: -0.2615869270302503, lo: -1.741997278446398e-17 },
        Df64 { hi: -0.25757741706362375, lo: -1.6107174092204261e-18 },
        Df64 { hi: -0.2535461358543676, lo: 7.096460077142018e-18 },
        Df64 { hi: -0.24949296518678724, lo: -4.31326076332226e-18 },
        Df64 { hi: -0.24541778620328863, lo: 4.688384843543075e-18 },
        Df64 { hi: -0.24132047940089266, lo: 6.212078255412209e-18 },
        Df64 { hi: -0.23720092462773085, lo: 3.8644266954502085e-19 },
        Df64 { hi: -0.233059001079522, lo: -1.1135017009065593e-17 },
        Df64 { hi: -0.2288945872960296, lo: 1.199359843285919e-17 },
        Df64 { hi: -0.2247075611575, lo: -7.300353295344693e-18 },
        Df64 { hi: -0.2204977998810815, lo: -8.849540348841276e-18 },
        Df64 { hi: -0.21626518001722356, lo: 3.750842387009219e-18 },
        Df64 { hi: -0.21200957744605675, lo: -5.068458235639152e-18 },
        Df64 { hi: -0.20773086737375313, lo: -9.668858517292851e-18 },
        Df64 { hi: -0.20342892432886656, lo: 5.039118519698011e-18 },
        Df64 { hi: -0.19910362215865332, lo: -2.5190116520100086e-18 },
        Df64 { hi: -0.19475483402537286, lo: 1.2353596284898944e-17 },
        Df64 { hi: -0.19038243240256814, lo: 1.0470667077114546e-17 },
        Df64 { hi: -0.1859862890713261, lo: -5.809199807906506e-18 },
        Df64 { hi: -0.18156627511651777, lo: 1.0736049740970466e-17 },
        Df64 { hi: -0.17712226092301758, lo: 4.882751662883964e-18 },
        Df64 { hi: -0.1726541161719028, lo: -7.294679715277685e-18 },
        Df64 { hi: -0.16816170983663178, lo: 1.699387867936586e-18 },
        Df64 { hi: -0.1636449101792017, lo: 3.719957926310978e-19 },
        Df64 { hi: -0.15910358474628547, lo: 1.3239474487278572e-17 },
        Df64 { hi: -0.15453760036534742, lo: 7.162793859283428e-18 },
        Df64 { hi: -0.14994682314073826, lo: -4.01185968519885e-18 },
        Df64 { hi: -0.1453311184497686, lo: 6.167253948093172e-18 },
        Df64 { hi: -0.14069035093876103, lo: -9.256902091315555e-18 },
        Df64 { hi: -0.13602438451908122, lo: 1.7562419252346148e-18 },
        Df64 { hi: -0.13133308236314686, lo: -1.1933629119164127e-17 },
        Df64 { hi: -0.12661630690041553, lo: 1.749698813720255e-18 },
        Df64 { hi: -0.12187391981335026, lo: 9.229156694299104e-19 },
        Df64 { hi: -0.1171057820333636, lo: 5.67321166697297e-18 },
        Df64 { hi: -0.11231175373673938, lo: 4.393083367153945e-18 },
        Df64 { hi: -0.1074916943405325, lo: -6.2125877472988e-18 },
        Df64 { hi: -0.1026454624984464, lo: -4.7640585938584126e-18 },
        Df64 { hi: -0.09777291609668806, lo: 1.869463571662324e-18 },
        Df64 { hi: -0.09287391224980063, lo: 5.66349353665608e-18 },
        Df64 { hi: -0.08794830729647335, lo: 4.713011919872412e-18 },
        Df64 { hi: -0.08299595679532877, lo: 2.537748313413679e-18 },
        Df64 { hi: -0.07801671552068704, lo: -1.94313451912091e-18 },
        Df64 { hi: -0.07301043745830721, lo: -6.701713777619857e-18 },
        Df64 { hi: -0.06797697580110548, lo: 4.948987787473942e-18 },
        Df64 { hi: -0.06291618294485005, lo: -2.8582414493917966e-18 },
        Df64 { hi: -0.057827910483832776, lo: 5.00397795774813e-19 },
        Df64 { hi: -0.05271200920651718, lo: 3.1392298682681924e-18 },
        Df64 { hi: -0.047568329091162896, lo: -2.025181945944751e-18 },
        Df64 { hi: -0.042396719301426355, lo: 2.4114209502780123e-18 },
        Df64 { hi: -0.037197028181937535, lo: -1.0025615211181075e-18 },
        Df64 { hi: -0.03196910325385278, lo: 3.089672476031033e-18 },
        Df64 { hi: -0.026712791210383356, lo: -6.393577718667539e-19 },
        Df64 { hi: -0.021427937912299865, lo: -2.989714202136461e-19 },
        Df64 { hi: -0.01611438838341211, lo: 4.670642216485574e-19 },
        Df64 { hi: -0.010771986806024515, lo: -6.223051570826017e-19 },
        Df64 { hi: -0.005400576516366824, lo: -2.342423707574178e-19 },
        Df64 { hi: 0.0, lo: 0.0 },
        Df64 { hi: 0.005429901112802822, lo: -4.1792582417406993e-19 },
        Df64 { hi: 0.01088928605170046, lo: 3.7773268042268547e-19 },
        Df64 { hi: 0.016378314910953037, lo: 1.2588974512148405e-18 },
        Df64 { hi: 0.02189714865411668, lo: -9.494539895697731e-19 },
        Df64 { hi: 0.027445949118763698, lo: -9.884844191031042e-19 },
        Df64 { hi: 0.03302487902122842, lo: 6.619449701198605e-19 },
        Df64 { hi: 0.03863410196137879, lo: -2.487307246639953e-18 },
        Df64 { hi: 0.04427378242741384, lo: 2.252170208492904e-18 },
        Df64 { hi: 0.049944085800687266, lo: 4.182272500122047e-19 },
        Df64 { hi: 0.05564517836055716, lo: 1.759325738772092e-18 },
        Df64 { hi: 0.06137722728926208, lo: 1.9042507224487988e-18 },
        Df64 { hi: 0.06714040067682361, lo: 4.268187178470922e-18 },
        Df64 { hi: 0.07293486752597556, lo: -3.839668843358824e-18 },
        Df64 { hi: 0.07876079775711979, lo: 2.8223346785063543e-18 },
        Df64 { hi: 0.08461836221330923, lo: 3.905952842534547e-18 },
        Df64 { hi: 0.09050773266525766, lo: -2.712245182495796e-18 },
        Df64 { hi: 0.09642908181637683, lo: -3.6881836132353304e-18 },
        Df64 { hi: 0.10238258330784095, lo: -2.8507825155508824e-18 },
        Df64 { hi: 0.10836841172367864, lo: -4.601411604918528e-18 },
        Df64 { hi: 0.11438674259589254, lo: -6.919517894059943e-18 },
        Df64 { hi: 0.12043775240960669, lo: -6.499707834283954e-18 },
        Df64 { hi: 0.1265216186082419, lo: -3.8525836433032604e-18 },
        Df64 { hi: 0.13263851959871922, lo: 4.617986051751087e-18 },
        Df64 { hi: 0.13878863475669165, lo: 5.861399913367335e-18 },
        Df64 { hi: 0.14497214443180423, lo: -9.09825230955772e-18 },
        Df64 { hi: 0.1511892299529827, lo: 4.751526573009359e-18 },
        Df64 { hi: 0.15744007363375104, lo: -7.971985464457258e-18 },
        Df64 { hi: 0.1637248587775775, lo: 1.0536472753612021e-17 },
        Df64 { hi: 0.1700437696832502, lo: -1.8477442017900047e-18 },
        Df64 { hi: 0.17639699165028128, lo: 3.088131092296112e-20 },
        Df64 { hi: 0.18278471098434104, lo: -1.2325821314838153e-17 },
        Df64 { hi: 0.18920711500272105, lo: 1.2064576699027549e-17 },
        Df64 { hi: 0.19566439203982738, lo: -9.345114526443012e-18 },
        Df64 { hi: 0.20215673145270313, lo: 1.0938663761265181e-17 },
        Df64 { hi: 0.20868432362658157, lo: 8.043891778967983e-18 },
        Df64 { hi: 0.21524735998046887, lo: 6.140419920071864e-18 },
        Df64 { hi: 0.2218460329727575, lo: 4.912090348488744e-18 },
        Df64 { hi: 0.22848053610687, lo: 8.767759302603614e-18 },
        Df64 { hi: 0.2351510639369333, lo: 3.469859019437239e-18 },
        Df64 { hi: 0.24185781207348406, lo: -8.930875312888462e-18 },
        Df64 { hi: 0.24860097718920474, lo: 6.4861685666710185e-19 },
        Df64 { hi: 0.2553807570246911, lo: -6.7113898212968784e-18 },
        Df64 { hi: 0.2621973503942507, lo: 2.4666502356519365e-17 },
        Df64 { hi: 0.2690509571917332, lo: 2.667932131342186e-18 },
        Df64 { hi: 0.2759417783963921, lo: -1.1868000020372746e-17 },
        Df64 { hi: 0.28287001607877826, lo: 1.713594918243561e-17 },
        Df64 { hi: 0.28983587340666583, lo: -2.1529727153539737e-17 },
        Df64 { hi: 0.29683955465100964, lo: 2.5382502794888315e-17 },
        Df64 { hi: 0.3038812651919359, lo: -2.4545546479836942e-17 },
        Df64 { hi: 0.31096121152476436, lo: -1.6304210123936712e-17 },
        Df64 { hi: 0.318079601266064, lo: 9.315929597662924e-19 },
        Df64 { hi: 0.32523664315974127, lo: 2.6923839130869213e-17 },
        Df64 { hi: 0.33243254708316144, lo: 4.495284922090389e-18 },
        Df64 { hi: 0.339667524053303, lo: -2.1749476514198334e-17 },
        Df64 { hi: 0.34694178623294586, lo: -2.3270500218711038e-17 },
        Df64 { hi: 0.3542555469368927, lo: 2.1498332566772065e-17 },
        Df64 { hi: 0.36160902063822475, lo: 1.533787661270668e-18 },
        Df64 { hi: 0.3690024229745906, lo: -1.5084323271327172e-17 },
        Df64 { hi: 0.3764359707545301, lo: -1.3474738127460185e-17 },
        Df64 { hi: 0.38390988196383197, lo: -1.2193965356690036e-17 },
        Df64 { hi: 0.3914243757719262, lo: 6.4494025783679345e-18 },
        Df64 { hi: 0.3989796725383111, lo: 1.4880170372002426e-17 },
        Df64 { hi: 0.40657599381901544, lo: 7.034914812136422e-18 }
        ];

    assert!(n >= -64 && n < 64);
    return EXPM1_ALPHAS[(n + 64) as usize];
}

#[cfg(test)]
mod test {
    use super::*;
    use super::super::checks::{is_finite, is_infinite, is_zero};
    use super::super::test_utils::*;
    use approx::assert_ulps_eq;
    use rug::ops::Pow;

    #[test]
    fn test_expm1_kernel()
    {
        // small values, start from ALPHA/2
        let mut x = Df64::from(0.0025);
        while x.hi > 1e-290 {
            check_unary(|x| expm1_kernel(x, 6, 10), |x| x.exp_m1(), x, 1.1);
            check_unary(|x| expm1_kernel(x, 6, 10), |x| x.exp_m1(), -x, 1.1);
            x *= 0.947;
        }
    }

    #[test]
    fn test_exp()
    {
        // special values
        assert!(is_infinite(exp(Df64::from(1000.0))));
        assert!(is_infinite(exp(Df64::INFINITY)));
        assert!(is_zero(exp(Df64::from(-1000.0))));
        assert!(is_zero(exp(-Df64::INFINITY)));
        assert!(is_nan(exp(Df64::NAN)));

        // simple vals
        check_unary(exp, |x| x.exp(), Df64::ZERO, 1.0);

        // small values
        let mut x = Df64::from(0.25);
        while x.hi > 1e-290 {
            check_unary(exp, |x| x.exp(), x, 1.0);
            check_unary(exp, |x| x.exp(), -x, 1.0);
            check_unary(exp2, |x| x.exp2(), x, 1.0);
            check_unary(exp2, |x| x.exp2(), -x, 1.0);
            x *= 0.947;
        }

        check_unary(exp, |x| x.exp(), Df64::ONE, 1.0);

        // large values
        x = Df64::from(0.25);
        while x.hi < LOG_MAX {
            check_unary(exp, |x| x.exp(), x, 1.0);
            if x.hi < 670.0 {
                check_unary(exp, |x| x.exp(), -x, 1.0);
            }

            // XXX
            check_unary(exp2, |x| x.exp2(), x, 1000.0);
            check_unary(exp2, |x| x.exp2(), -x, 1000.0);
            x *= 1.0041;
        }

        check_unary(exp, |x| x.exp(), Df64::from(LOG_MAX), 1.0);
        assert!(is_finite(exp(Df64::from(LOG_MAX))));
        assert!(is_infinite(exp(Df64::from((1.0 + f64::EPSILON) * LOG_MAX))));
    }

    #[test]
    fn test_expm1()
    {
        // special values
        assert!(is_infinite(expm1(Df64::from(1000.0))));
        assert!(is_infinite(expm1(Df64::INFINITY)));
        assert!(expm1(Df64::from(-1000.0)) == Df64::from(-1.0));
        assert!(expm1(-Df64::INFINITY) == Df64::from(-1.0));
        assert!(is_nan(expm1(Df64::NAN)));

        // simple vals
        check_unary(expm1, |x| x.exp_m1(), Df64::ZERO, 1.0);
        check_unary(expm1, |x| x.exp_m1(), Df64::ONE, 1.0);

        // small values
        // XXX here we have to work on the kernel
        let mut x = Df64::from(0.5);
        while x.hi > 1e-290 {
            check_unary(expm1, |x| x.exp_m1(), x, 1.5);
            check_unary(expm1, |x| x.exp_m1(), -x, 1.5);
            x *= 0.947;
        }

        // large values
        x = Df64::from(0.5);
        while x.hi < LOG_MAX {
            check_unary(expm1, |x| x.exp_m1(), x, 1.0);
            if x.hi < 670.0 {
                check_unary(expm1, |x| x.exp_m1(), -x, 1.0);
            }
            x *= 1.0041;
        }
    }

    #[test]
    fn test_log()
    {
        // special values
        assert!(is_infinite(log(Df64::INFINITY)));
        assert!(is_infinite(log(Df64::ZERO)));
        assert!(is_nan(log(Df64::from(-0.1))));
        assert!(is_nan(log(Df64::NEG_INFINITY)));
        assert!(is_nan(log(Df64::NAN)));

        // simple vals
        check_unary(log, |x| x.ln(), Df64::ONE, 1.0);
        check_unary(log, |x| x.ln(), Df64::from(3.0), 1.0);

        // small values
        let mut x = Df64::ONE;
        while x.hi > 1e-290 {
            check_unary(log, |x| x.ln(), x, 1.0);
            check_unary(log2, |x| x.log2(), x, 3.0);
            check_unary(log10, |x| x.log10(), x, 3.0);
            x *= 0.947;
        }

        // large values
        x = Df64::ONE;
        while x.hi < 1e300 {
            check_unary(log, |x| x.ln(), x, 1.0);
            check_unary(log2, |x| x.log2(), x, 1.0);
            check_unary(log10, |x| x.log10(), x, 1.0);
            x *= 1.13;
        }
    }

    #[test]
    fn test_log1p()
    {
        // special values
        assert!(is_infinite(log1p(Df64::INFINITY)));
        assert!(is_infinite(log1p(Df64::from(-1.0))));
        assert!(is_nan(log1p(Df64::from(-1.1))));
        assert!(is_nan(log1p(Df64::NEG_INFINITY)));
        assert!(is_nan(log1p(Df64::NAN)));

        // simple vals
        check_unary(log1p, |x| x.ln_1p(), Df64::ZERO, 0.0);
        check_unary(log1p, |x| x.ln_1p(), Df64::ONE, 1.5);

        // small values
        let mut x = Df64::from(0.99);
        while x.hi > 1e-290 {
            check_unary(log1p, |x| x.ln_1p(), x, 1.5);
            check_unary(log1p, |x| x.ln_1p(), -x, 1.5);
            x *= 0.947;
        }

        // large values
        x = Df64::ONE;
        while x.hi < 1e300 {
            check_unary(log1p, |x| x.ln_1p(), x, 1.0);
            x *= 1.13;
        }
    }

    #[test]
    fn test_powf()
    {
        let two = Df64::from(2.0);
        let neg_two = Df64::from(-2.0);
        let neg_zero = Df64::from(-0.0);

        // x^0 == 1 and 1^y == 1, also for NaN arguments
        assert!(powf(Df64::NAN, Df64::ZERO) == Df64::ONE);
        assert!(powf(Df64::INFINITY, Df64::ZERO) == Df64::ONE);
        assert!(powf(Df64::ZERO, Df64::ZERO) == Df64::ONE);
        assert!(powf(Df64::ONE, Df64::NAN) == Df64::ONE);
        assert!(powf(Df64::ONE, Df64::INFINITY) == Df64::ONE);

        // NaN propagation
        assert!(is_nan(powf(Df64::NAN, two)));
        assert!(is_nan(powf(two, Df64::NAN)));
        assert!(is_nan(powf(neg_two, Df64::NAN)));

        // zero and infinite base, including signed zero
        assert!(powf(Df64::ZERO, two) == Df64::ZERO);
        assert!(is_infinite(powf(Df64::ZERO, Df64::from(-1.0))));
        assert!(powf(neg_zero, Df64::from(3.0)).hi.is_sign_negative());
        assert!(powf(neg_zero, Df64::from(2.0)).hi.is_sign_positive());
        assert!(powf(neg_zero, Df64::from(-3.0)) == Df64::NEG_INFINITY);
        assert!(powf(neg_zero, Df64::from(-2.0)) == Df64::INFINITY);
        assert!(powf(Df64::INFINITY, two) == Df64::INFINITY);
        assert!(powf(Df64::INFINITY, Df64::from(-2.0)) == Df64::ZERO);
        assert!(powf(Df64::NEG_INFINITY, Df64::from(3.0)) == Df64::NEG_INFINITY);
        assert!(powf(Df64::NEG_INFINITY, Df64::from(2.0)) == Df64::INFINITY);
        assert!(powf(Df64::NEG_INFINITY, Df64::from(-3.0)).hi.is_sign_negative());
        assert!(powf(Df64::NEG_INFINITY, Df64::from(-2.0)) == Df64::ZERO);

        // ... also for the unit exponents
        assert!(powf(Df64::ZERO, Df64::ONE) == Df64::ZERO);
        assert!(is_infinite(powf(Df64::ZERO, -Df64::ONE)));
        assert!(powf(neg_zero, Df64::ONE).hi.is_sign_negative());
        assert!(powf(neg_zero, -Df64::ONE) == Df64::NEG_INFINITY);
        assert!(powf(Df64::INFINITY, Df64::ONE) == Df64::INFINITY);
        assert!(powf(Df64::INFINITY, -Df64::ONE) == Df64::ZERO);
        assert!(powf(Df64::NEG_INFINITY, Df64::ONE) == Df64::NEG_INFINITY);
        assert!(powf(Df64::NEG_INFINITY, -Df64::ONE) == Df64::from(-0.0));

        // The negative base path must not turn an overflow into NaN either
        assert!(is_infinite(powf(Df64::from(2.0), Df64::from(2000.0))));
        assert!(is_infinite(powf(Df64::from(-2.0), Df64::from(2000.0))));
        assert!(powf(Df64::from(-2.0), Df64::from(2001.0)) == Df64::NEG_INFINITY);
        assert!(powf(Df64::from(-2.0), Df64::from(1024.0)).hi.is_sign_positive());
        assert!(powf(Df64::from(2.0), Df64::from(-2000.0)) == Df64::ZERO);
        assert!(powf(Df64::from(-2.0), Df64::from(-2000.0)) == Df64::ZERO);

        // infinite exponent
        assert!(powf(Df64::from(0.5), Df64::INFINITY) == Df64::ZERO);
        assert!(powf(Df64::from(0.5), Df64::NEG_INFINITY) == Df64::INFINITY);
        assert!(powf(two, Df64::INFINITY) == Df64::INFINITY);
        assert!(powf(two, Df64::NEG_INFINITY) == Df64::ZERO);
        assert!(powf(-Df64::ONE, Df64::INFINITY) == Df64::ONE);
        assert!(powf(-Df64::ONE, Df64::NEG_INFINITY) == Df64::ONE);

        // negative base: NaN for a non-integer, sign for an integer exponent
        assert!(is_nan(powf(neg_two, Df64::from(0.5))));
        assert_ulps_eq!(powf(neg_two, Df64::from(3.0)), Df64::from(-8.0), max_ulps = 8);
        assert_ulps_eq!(powf(neg_two, Df64::from(2.0)), Df64::from(4.0), max_ulps = 8);
        assert_ulps_eq!(powf(neg_two, Df64::from(-1.0)), Df64::from(-0.5), max_ulps = 8);

        // precision against the multiprecision reference.  The exponentiation
        // is carried out as exp(y * log(x)), so the error grows with
        // |y * log(x)| (up to ~7.5 here).
        let mut x = Df64::from(0.125);
        while x.hi < 8.0 {
            check_binary(powf, |x, y| x.pow(y), x, Df64::from(2.5), 16.0);
            check_binary(powf, |x, y| x.pow(y), x, Df64::from(-1.5), 16.0);
            check_binary(powf, |x, y| x.pow(y), Df64::from(2.5), x, 16.0);
            x *= 1.13;
        }
    }

    /// `powi` with a fixed exponent, so that it can be checked with
    /// `check_binary` against the multiprecision reference.
    fn powi_three(base: Df64, _expo: Df64) -> Df64
    {
        return powi(base, 3);
    }

    #[test]
    fn test_powi()
    {
        let two = Df64::from(2.0);

        // exact cases
        assert!(powi(Df64::NAN, 0) == Df64::ONE);
        assert!(powi(Df64::ZERO, 0) == Df64::ONE);
        assert!(powi(two, 1) == two);
        assert!(powi(two, -1) == Df64::from(0.5));
        assert!(powi(Df64::from(-2.0), 1) == Df64::from(-2.0));
        assert!(powi(Df64::from(-2.0), -1) == Df64::from(-0.5));

        // NaN propagation
        assert!(is_nan(powi(Df64::NAN, 3)));

        // zero and infinite base, including signed zero
        assert!(powi(Df64::ZERO, 3) == Df64::ZERO);
        assert!(is_infinite(powi(Df64::ZERO, -3)));
        assert!(powi(Df64::from(-0.0), 3).hi.is_sign_negative());
        assert!(powi(Df64::from(-0.0), 2).hi.is_sign_positive());
        assert!(powi(Df64::from(-0.0), -3) == Df64::NEG_INFINITY);
        assert!(powi(Df64::INFINITY, 3) == Df64::INFINITY);
        assert!(powi(Df64::INFINITY, -3) == Df64::ZERO);
        assert!(powi(Df64::NEG_INFINITY, 3) == Df64::NEG_INFINITY);
        assert!(powi(Df64::NEG_INFINITY, 4) == Df64::INFINITY);
        assert!(powi(Df64::NEG_INFINITY, -3).hi.is_sign_negative());
        assert!(powi(Df64::NEG_INFINITY, -4) == Df64::ZERO);

        // The unit exponents take a short cut and therefore must not bypass
        // the zero and infinity handling above.
        assert!(powi(Df64::ZERO, 1) == Df64::ZERO);
        assert!(is_infinite(powi(Df64::ZERO, -1)));
        assert!(powi(Df64::from(-0.0), 1).hi.is_sign_negative());
        assert!(powi(Df64::from(-0.0), -1) == Df64::NEG_INFINITY);
        assert!(powi(Df64::INFINITY, 1) == Df64::INFINITY);
        assert!(powi(Df64::INFINITY, -1) == Df64::ZERO);
        assert!(powi(Df64::NEG_INFINITY, 1) == Df64::NEG_INFINITY);
        assert!(powi(Df64::NEG_INFINITY, -1) == Df64::from(-0.0));
        assert!(is_nan(powi(Df64::NAN, 1)));
        assert!(is_nan(powi(Df64::NAN, -1)));
        assert!(powi(Df64::ONE, 1) == Df64::ONE);
        assert!(powi(Df64::ONE, -1) == Df64::ONE);

        // Overflow has to reach infinity rather than NaN: the sign is applied
        // by negation because `1.0 * Df64::INFINITY` is NaN.
        assert!(is_infinite(powi(Df64::from(2.0), 1024)));
        assert!(is_infinite(powi(Df64::from(2.0), 2000)));
        assert!(is_infinite(powi(Df64::from(-2.0), 2000)));
        assert!(powi(Df64::from(-2.0), 2001) == Df64::NEG_INFINITY);
        assert!(is_infinite(powi(Df64::from(0.5), -2000)));
        assert!(powi(Df64::from(0.5), 2000) == Df64::ZERO);
        assert!(powi(Df64::from(2.0), -2000) == Df64::ZERO);
        assert!(powi(Df64::from(-0.5), i32::MAX) == Df64::from(-0.0));
        assert!(is_infinite(powi(Df64::from(-0.5), i32::MIN)));

        // negative base with an odd and an even exponent
        assert_ulps_eq!(powi(Df64::from(-2.0), 3), Df64::from(-8.0), max_ulps = 8);
        assert_ulps_eq!(powi(Df64::from(-2.0), 2), Df64::from(4.0), max_ulps = 8);

        // precision against the multiprecision reference
        let mut x = Df64::from(0.125);
        while x.hi < 8.0 {
            check_binary(powi_three, |x, y| x.pow(y), x, Df64::from(3.0), 16.0);
            check_binary(powi_three, |x, y| x.pow(y), -x, Df64::from(3.0), 16.0);
            x *= 1.13;
        }
    }
}
