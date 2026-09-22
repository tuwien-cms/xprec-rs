use crate::Df64;
use crate::arith;
use core::convert::TryFrom;
use num_traits;

/// Error returned when a `Df64` does not fit into an integer type.
///
/// This is the error type of the `TryFrom<Df64>` implementations.  The same
/// failure is reported as `None` by the `num_traits::ToPrimitive` methods and
/// by the `try_to_*` functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TryFromDf64Error
{
    /// The value that is not representable in the target type.
    pub value: Df64,
}

impl core::fmt::Display for TryFromDf64Error
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
    {
        return write!(
            f, "{} is out of range for the requested integer type", self.value);
    }
}

impl std::error::Error for TryFromDf64Error { }

macro_rules! convert {
    ($fname:ident Df64 $Dest:ident) => {
        /// Convert Df64 to $Dest, if target in range
        ///
        /// Converts `x` to integer type $Dest, if `x` is indeed in the range
        /// `$Dest::MIN <= x < $Dest::MAX + 1`. In converting, any fractional
        /// part is discarded.
        #[inline]
        pub fn $fname(x: Df64) -> Option<$Dest>
        {
            const LOW: f64 = $Dest::MIN as f64;
            const HIGH: f64 = ($Dest::MAX as f64) + 1.0;
            if x.hi >= LOW && x.hi < HIGH {
                let xi = unsafe {
                    x.hi.to_int_unchecked::<$Dest>() +
                    x.lo.to_int_unchecked::<$Dest>()
                };
                return Some(xi);
            } else {
                return None;
            }
        }

        impl TryFrom<Df64> for $Dest
        {
            type Error = TryFromDf64Error;

            #[inline]
            fn try_from(x: Df64) -> Result<$Dest, TryFromDf64Error>
            {
                return $fname(x).ok_or(TryFromDf64Error { value: x });
            }
        }
    };
    ($fname:ident $Src:ident Df64) => {
        /// Convert $Src to Df64
        ///
        /// Converts the integer `x` of type $Src to compensated float. Takes
        /// care to preserve the least significant bits of i64, which are
        /// typically truncated in f64.
        #[inline]
        pub fn $fname(x: $Src) -> Df64
        {
            const SIZEOF_SRC: usize = size_of::<$Src>();
            if SIZEOF_SRC == 8 {
                const HI_HALF: $Src = !(0 as $Src) << (4 * SIZEOF_SRC);
                const LO_HALF: $Src = !HI_HALF;
                return arith::addfast_dd(
                            (x & HI_HALF) as f64, (x & LO_HALF) as f64);
            } else {
                return Df64 {hi: x as f64, lo: 0.0};
            }
        }

        impl From<$Src> for Df64
        {
            #[inline]
            fn from(x: $Src) -> Df64
            {
                return $fname(x);
            }
        }
    };
}

// Instantiate all conversion routines

convert!(try_to_isize Df64   isize);
convert!(try_to_i8    Df64   i8);
convert!(try_to_i16   Df64   i16);
convert!(try_to_i32   Df64   i32);
convert!(try_to_i64   Df64   i64);
convert!(try_to_i128  Df64   i128);

convert!(try_to_usize Df64   usize);
convert!(try_to_u8    Df64   u8);
convert!(try_to_u16   Df64   u16);
convert!(try_to_u32   Df64   u32);
convert!(try_to_u64   Df64   u64);
convert!(try_to_u128  Df64   u128);

// XXX add u128/i128

convert!(from_isize   isize Df64);
convert!(from_i8      i8    Df64);
convert!(from_i16     i16   Df64);
convert!(from_i32     i32   Df64);
convert!(from_i64     i64   Df64);

convert!(from_usize   usize Df64);
convert!(from_u8      u8    Df64);
convert!(from_u16     u16   Df64);
convert!(from_u32     u32   Df64);
convert!(from_u64     u64   Df64);

// TRAITS

impl From<Df64> for f64 {
    fn from(src: Df64) -> f64 {
        src.hi
    }
}

impl From<f64> for Df64 {
    fn from(src: f64) -> Df64 {
        Df64 {hi: src, lo: 0.0}
    }
}


impl num_traits::ToPrimitive for Df64 {
    #[inline] fn to_isize(&self) -> Option<isize> { return try_to_isize(*self); }
    #[inline] fn to_i8(&self)    -> Option<i8>    { return try_to_i8(*self); }
    #[inline] fn to_i16(&self)   -> Option<i16>   { return try_to_i16(*self); }
    #[inline] fn to_i32(&self)   -> Option<i32>   { return try_to_i32(*self); }
    #[inline] fn to_i64(&self)   -> Option<i64>   { return try_to_i64(*self); }
    #[inline] fn to_i128(&self)  -> Option<i128>  { return try_to_i128(*self); }

    #[inline] fn to_usize(&self) -> Option<usize> { return try_to_usize(*self); }
    #[inline] fn to_u8(&self)    -> Option<u8>    { return try_to_u8(*self); }
    #[inline] fn to_u16(&self)   -> Option<u16>   { return try_to_u16(*self); }
    #[inline] fn to_u32(&self)   -> Option<u32>   { return try_to_u32(*self); }
    #[inline] fn to_u64(&self)   -> Option<u64>   { return try_to_u64(*self); }
    #[inline] fn to_u128(&self)  -> Option<u128>  { return try_to_u128(*self); }

    #[inline] fn to_f32(&self)   -> Option<f32>   { return Some(self.hi as f32); }
    #[inline] fn to_f64(&self)   -> Option<f64>   { return Some(self.hi); }
}

impl num_traits::FromPrimitive for Df64 {
    #[inline] fn from_isize(n: isize) -> Option<Df64> { return Some(from_isize(n)); }
    #[inline] fn from_i8(n: i8)       -> Option<Df64> { return Some(from_i8(n)); }
    #[inline] fn from_i16(n: i16)     -> Option<Df64> { return Some(from_i16(n)); }
    #[inline] fn from_i32(n: i32)     -> Option<Df64> { return Some(from_i32(n)); }
    #[inline] fn from_i64(n: i64)     -> Option<Df64> { return Some(from_i64(n)); }

    #[inline] fn from_usize(n: usize) -> Option<Df64> { return Some(from_usize(n)); }
    #[inline] fn from_u8(n: u8)       -> Option<Df64> { return Some(from_u8(n)); }
    #[inline] fn from_u16(n: u16)     -> Option<Df64> { return Some(from_u16(n)); }
    #[inline] fn from_u32(n: u32)     -> Option<Df64> { return Some(from_u32(n)); }
    #[inline] fn from_u64(n: u64)     -> Option<Df64> { return Some(from_u64(n)); }

    #[inline] fn from_f32(n: f32)     -> Option<Df64> { return Some(Df64::from(n as f64)); }
    #[inline] fn from_f64(n: f64)     -> Option<Df64> { return Some(Df64::from(n)); }
}

impl num_traits::NumCast for Df64 {
    /// Generic numeric cast via f64 intermediate representation.
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        num_traits::FromPrimitive::from_f64(n.to_f64()?)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn test_from_primitive_f64() {
        // Test basic f64 conversion
        let val = 1.5;
        let df64 = Df64::from_f64(val).unwrap();
        assert_eq!(df64.hi, 1.5);
        assert_eq!(df64.lo, 0.0);
    }

    #[test]
    fn test_from_primitive_f32() {
        // Test f32 conversion
        let val = 2.25f32;
        let df64 = Df64::from_f32(val).unwrap();
        assert_eq!(df64.hi, 2.25);
        assert_eq!(df64.lo, 0.0);
    }

    #[test]
    fn test_from_primitive_integers() {
        // Test integer conversions
        assert_eq!(Df64::from_i32(42).unwrap(), Df64 { hi: 42.0, lo: 0.0 });
        assert_eq!(Df64::from_u64(100).unwrap(), Df64 { hi: 100.0, lo: 0.0 });
        assert_eq!(Df64::from_isize(-5).unwrap(), Df64 { hi: -5.0, lo: 0.0 });
    }

    #[test]
    fn test_from_primitive_edge_cases() {
        // Test edge cases
        assert_eq!(Df64::from_f64(0.0).unwrap(), Df64::ZERO);
        assert_eq!(Df64::from_f64(1.0).unwrap(), Df64::ONE);
        assert_eq!(Df64::from_f64(-1.0).unwrap(), Df64 { hi: -1.0, lo: 0.0 });
        
        // Test very small numbers
        let small = 1e-10;
        let df64 = Df64::from_f64(small).unwrap();
        assert_eq!(df64.hi, small);
        assert_eq!(df64.lo, 0.0);
    }
}
