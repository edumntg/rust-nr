mod models;
mod enums;
mod loaders;
mod solvers;

use std::time::Instant;
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
    
    // Time the solving process
    let start_time = Instant::now();
    let stats = solver.solve(&mut x);
    let duration = start_time.elapsed();
    
    if stats.converged {
        println!("\nConvergence Status:");
        println!("--------------------------------------------------");
        println!("Status:      Converged!");
        println!("Iterations:  {}", stats.iterations);
        println!("Final Error: {:.10}", stats.error);
        println!("Solve Time:  {:?}", duration);
        println!("--------------------------------------------------");

        power_system.print_results(&x);
    } else {
        println!("\nPower Flow Failed to Converge.");
        println!("--------------------------------------------------");
        println!("Status:      Failed");
        println!("Iterations:  {}", stats.iterations);
        println!("Last Error:  {:.10}", stats.error);
        println!("Time:        {:?}", duration);
        println!("--------------------------------------------------");
    }
}
