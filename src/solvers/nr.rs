use ndarray::prelude::*;
use ndarray_linalg::solve::Solve;

pub struct SolveStats {
    pub converged: bool,
    pub iterations: i32,
    pub error: f64,
}

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

    pub fn solve(&self, x: &mut Array2<f64>) -> SolveStats {
        let mut err = 1.0e9;
        let mut current_iter = 0;

        // Loop until convergence or maximum iterations reached
        while err > self.tol && current_iter < self.max_iters {
            let x_old = x.clone();

            let fx = (self.f)(x);
            let dfx = (self.df)(x); 

            // Convert fx to a 1D array as required by the solve method
            let fx_1d = fx.clone().into_shape((fx.len(),)).unwrap();

            // Solve J * delta_x = f(x) using LU decomposition
            let step_1d = match dfx.solve(&fx_1d) {
                Ok(s) => s,
                Err(_) => {
                    return SolveStats {
                        converged: false,
                        iterations: current_iter,
                        error: err,
                    };
                }
            };

            // Convert the 1D step back to a 2D column vector for updating x
            let step = step_1d.into_shape((fx.len(), 1)).unwrap();

            // Update state with the calculated step
            *x -= &step;

            // Use the infinity norm (max absolute difference) to check convergence
            let diff = x.clone() - x_old;
            err = diff.mapv(f64::abs).fold(0.0, |a, b| a.max(*b));
            current_iter += 1;
            println!("Iteration {} | Max Mismatch: {:.8}", current_iter, err);
        }
        
        SolveStats {
            converged: err <= self.tol,
            iterations: current_iter,
            error: err,
        }
    }
}
