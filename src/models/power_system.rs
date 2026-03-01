use ndarray::Array2;
use num_complex::Complex;
use crate::models::bus::Bus;
use crate::models::line::Line;

pub struct PowerSystem {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub ybus: Array2<Complex<f64>>,
}

impl PowerSystem {
    pub fn new(buses: Vec<Bus>, lines: Vec<Line>) -> Self {
        let ybus = PowerSystem::build_ybus(&buses, &lines);
        PowerSystem { buses, lines, ybus }
    }

    pub fn build_ybus(buses: &Vec<Bus>, lines: &Vec<Line>) -> Array2<Complex<f64>> {
        let n = buses.len();
        let mut ybus = Array2::<Complex<f64>>::zeros((n, n));

        for line in lines {
            let y = 1.0 / line.z();
            let from_idx = line.from;
            let to_idx = line.to;

            ybus[[from_idx, from_idx]] += y + Complex::new(0.0, line.b / 2.0);
            ybus[[to_idx, to_idx]] += y + Complex::new(0.0, line.b / 2.0);
            ybus[[from_idx, to_idx]] -= y;
            ybus[[to_idx, from_idx]] -= y;
        }

        ybus
    }

    pub fn print(&self) {
        println!("Power System Summary");
        println!("====================");
        
        println!("\nBuses:");
        println!("{:<5} {:<15} {:<8} {:<8} {:<8} {:<8} {:<10}", "ID", "Name", "V (pu)", "Theta", "P Load", "Q Load", "Type");
        println!("{}", "-".repeat(70));
        for bus in &self.buses {
            println!(
                "{:<5} {:<15} {:<8.3} {:<8.3} {:<8.3} {:<8.3} {:?}",
                bus.id, bus.name, bus.v, bus.theta, bus.p_load, bus.q_load, bus.bus_type
            );
        }

        println!("\nLines:");
        println!("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}", "From", "To", "R", "X", "B", "Status");
        println!("{}", "-".repeat(60));
        for line in &self.lines {
            println!(
                "{:<10} {:<10} {:<10.4} {:<10.4} {:<10.4} {}",
                line.from, line.to, line.r, line.x, line.b, if line.online { "Online" } else { "Offline" }
            );
        }
        println!("\nY-Bus Dimensions: {}x{}", self.ybus.nrows(), self.ybus.ncols());
    }
}
