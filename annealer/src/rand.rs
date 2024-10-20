pub const MAX: f64 = std::u64::MAX as f64;
use std::num::Wrapping;

#[inline]
fn rotl(x: u64, k: i32) -> u64 {
    (x << k) | (x >> (64 - k))
}

#[derive(Debug)]
pub struct UniformDist([u64; 2]);
impl UniformDist {
    pub fn next(&mut self) -> u64 {
        let s0 = Wrapping(self.0[0]);
        let mut s1 = Wrapping(self.0[1]);
        let result = s0 + s1;
        s1 ^= s0;
        self.0[0] = (Wrapping(rotl(s0.0, 55)) ^ s1 ^ (s1 << 14)).0;
        self.0[1] = rotl(s1.0, 36);
        result.0
    }

    pub fn choice<T: Clone>(&mut self, v: &Vec<T>) -> T {
        v[self.next() as usize % v.len()].clone()
    }

    pub fn new(seed: [u64; 2]) -> UniformDist {
        Self([seed[0], seed[1]])
    }
}
