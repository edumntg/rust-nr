use ndarray::prelude::*;
use ndarray_linalg::solve::Inverse;

// Generic Newton-Raphson solver
pub struct NewtonRaphson {
    pub f: Box<dyn Fn(&Array2<f64>) -> Array2<f64>>,
    pub df: Box<dyn Fn(&Array2<f64>) -> Array2<f64>>,
    pub tol: f64,
    pub max_iters: i32,
}

impl NewtonRaphson {
    pub fn new(
        f: Box<dyn Fn(&Array2<f64>) -> Array2<f64>>,
        df: Box<dyn Fn(&Array2<f64>) -> Array2<f64>>,
        tol: Option<f64>,
        max_iters: Option<i32>,
    ) -> Self {
        NewtonRaphson {
            f,
            df,
            tol: tol.unwrap_or(1e-6),
            max_iters: max_iters.unwrap_or(20),
        }
    }

    pub fn solve(&self, x: &mut Array2<f64>) -> bool {
        let mut err = 1.0e9;
        let mut current_iter = 0;
        let mut x_old: Array2<f64>;

        // Solving loop
        while err > self.tol && current_iter < self.max_iters {
            x_old = x.clone();

            let fx = (self.f)(x);
            let dfx = (self.df)(x); // Jacobian

            // Inverse of Jacobian
            let j_inv = match dfx.inv() {
                Ok(inv) => inv,
                Err(_) => {
                    println!("Jacobian is singular!");
                    return false;
                }
            };

            // NR Step: x_new = x_old - J^-1 * f(x)
            let step = j_inv.dot(&fx);

            // Update x
            *x -= &step;

            // Calculate error
            let diff = x.clone() - x_old;
            err = diff.mapv(f64::abs).fold(0.0, |a, b| a.max(*b));
            current_iter += 1;
            println!("Iter {} Err {:.8}", current_iter, err);
        }
        
        err <= self.tol
    }
}
