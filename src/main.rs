use ndarray::prelude::*;
use ndarray_linalg::solve::Inverse;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
// Import the trait

// Hello word for newton-raphson using rust

fn fx(x: &Array2<f64>) -> Array2<f64> {
    // Return a vector of 2x1 with the two functions
    ndarray::array![
        [x[[0,0]]*x[[0,0]] + x[[1,0]]*x[[1,0]] - 5.0], // f1(x,y) = x^2 + y^2 - 5 = 0
        [x[[0,0]]*x[[0,0]]*x[[0,0]] + x[[1,0]]*x[[1,0]]*x[[1,0]] - 2.0] // f2(x,y) = x^3 + y^3 - 2 = 0
    ] // (2,1)
}

fn dfx(x: &Array2<f64>) -> Array2<f64> {
    // Return a vector of 2x1 with the two functions

    ndarray::array![
        [2.0*x[[0,0]], 2.0*x[[1,0]]], // df1/dx, df1/dy
        [3.0*x[[0,0]]*x[[0,0]], 3.0*x[[1,0]]*x[[1,0]]] // df2/dx, df2/dy
    ] // (2,2)
}

// Now let's declare here a more generic version
fn solve(
    f: fn(&Array2<f64>) -> Array2<f64>,
    df: fn(&Array2<f64>) -> Array2<f64>,
    x: &mut Array2<f64>,
    tol_: Option<f64>,
    max_iters_: Option<i32>
) {
    let tol = tol_.unwrap_or(1e-3);
    let max_iters = max_iters_.unwrap_or(10);

    let mut err = 1.0e9;
    let mut current_iter = 0;
    let mut x_old;

    // Solving loop
    while err > tol && current_iter < max_iters {
        x_old = x.clone();

        let fx = f(&x);
        let dfx = df(&x); // Jacobian

        // Inverse of Jacobian
        let j_inv = &dfx.inv().unwrap();

        // NR Step
        let step = j_inv.dot(&fx);

        // Update x
        *x -= &step;

        // Calculate error
        let diff = &*x - &x_old;
        err = diff.mapv(f64::abs).fold(0.0, |a,b| a.max(*b));
        current_iter += 1;
        println!("Iter {} Err {:.8}, x = {:.4}, y = {:.4}", current_iter, err, x[[0,0]], x[[1,0]]);
    }
}

fn main() {
    let tol: f64 = 1.0e-9;
    let max_iters: i32 = 20;

    let mut x: Array2<f64> = ndarray::array![
        [2.0],
        [-1.0]
    ];

    solve(
        fx,
        dfx,
        &mut x,
        Option::from(tol),
        Option::from(max_iters)
    );

    println!("Sol = {:.4}", x);



    // let mut x_old;
    //
    // // Solving loop
    // while err > tol && current_iters < max_iters {
    //     x_old = x.clone();
    //
    //     let fx = f(&x);
    //     let dfx = df(&x); // Jacobian
    //
    //     // Inverse of Jacobian
    //     let j_inv = &dfx.inv().unwrap();
    //
    //     // NR Step
    //     let step = j_inv.dot(&fx);
    //
    //     // Update x
    //     x = &x - &step;
    //
    //     // Calculate error
    //     let diff = &x - &x_old;
    //     err = diff.mapv(f64::abs).fold(0.0, |a,b| a.max(*b));
    //     current_iters += 1;
    //     println!("Iter {} Err {:.8}, x = {:.4}, y = {:.4}", current_iters, err, x[[0,0]], x[[1,0]]);
    // }
}
