mod models;
mod enums;
mod loaders;
mod solvers;

use crate::loaders::data_loaders::load_ieee;
use crate::models::power_system::PowerSystem;
use crate::solvers::nr::NewtonRaphson;

fn main() {
    // Load IEEE-14 bus system
    let power_system: PowerSystem = load_ieee("src/datasets/ieee14.txt");
    power_system.print();

    println!("\nStarting Power Flow Analysis (Newton-Raphson)...");
    
    let (f, df) = power_system.prepare();
    let mut x = power_system.initial_x();
    
    let solver = NewtonRaphson::new(f, df, Some(1e-6), Some(20));
    
    if solver.solve(&mut x) {
        println!("\nPower Flow Converged!");
        power_system.print_results(&x);
    } else {
        println!("\nPower Flow Failed to Converge.");
    }
}
