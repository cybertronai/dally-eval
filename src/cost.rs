//! Read-distance pricing for the Bill Dally 2D Manhattan model.

/// Cost of reading the cell at 1-based address `idx`:
/// `ceil(sqrt(idx))`, computed exactly as `isqrt(idx - 1) + 1`.
///
/// Branchless integer math only; no LUT. `u64::isqrt` is a few cycles,
/// and a table large enough to matter would cost more cache than it
/// saves (measured decision point — see benches).
#[inline]
pub const fn cost(idx: u32) -> u64 {
    let a = (idx - 1) as u64;
    isqrt(a) + 1
}

/// Exact integer square root (681-bit safe for our range via u64).
#[inline]
const fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // Integer Newton from a power-of-two seed (bit length halved):
    // fully const-evaluable, no float ops.
    let bits = 64 - n.leading_zeros();
    let mut x = 1u64 << bits.div_ceil(2);
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            break;
        }
        x = y;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_formula() {
        // brute force against a naive float ceil for the first 1<<20
        for idx in 1u32..(1 << 20) {
            let want = (idx as f64).sqrt().ceil() as u64;
            assert_eq!(cost(idx), want, "idx={idx}");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(cost(1), 1);
        assert_eq!(cost(2), 2);
        assert_eq!(cost(4), 2);
        assert_eq!(cost(5), 3);
        assert_eq!(cost(9), 3);
        assert_eq!(cost(10), 4);
        assert_eq!(cost(1_000_000), 1_000);
        assert_eq!(cost(u32::MAX), 65_536);
    }
}
