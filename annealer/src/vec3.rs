use std::ops::{Add, Mul, Sub};

pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self
    }
}

impl<'a> Add<&'a Vector3> for &'a Vector3 {
    type Output = Vector3;
    fn add(self, rhs: Self) -> Self::Output {
        Vector3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<'a> Sub<&'a Vector3> for &'a Vector3 {
    type Output = Vector3;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Vector3 {
    pub fn cross(&self, rhs: &Vector3) -> Vector3 {
        Vector3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.z * rhs.y,
        }
    }

    pub fn normalize(mut self) -> Vector3 {
        let len = self.len();
        self.x /= len;
        self.y /= len;
        self.z /= len;
        self
    }

    pub fn len(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}
