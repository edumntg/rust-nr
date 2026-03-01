use num_complex::Complex;

#[derive(Debug, Clone)]
pub struct Line {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b: f64,
    pub tap: f64,
    pub shift: f64,
    pub online: bool,
}

impl Line {
    pub fn new(from: i32, to: i32, r: f64, x: f64, b: f64, tap: f64, shift: f64, online: bool) -> Self {
        Line {
            from: from as usize,
            to: to as usize,
            r,
            x,
            b,
            tap,
            shift,
            online,
        }
    }

    pub fn z(&self) -> Complex<f64> {
        Complex::new(self.r, self.x)
    }
}
