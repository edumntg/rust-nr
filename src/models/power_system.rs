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

        // Line and Transformer contributions
        for line in lines {
            if !line.online {
                continue;
            }

            let y_series = 1.0 / line.z();
            let y_shunt = Complex::new(0.0, line.b / 2.0);
            
            let from_idx = line.from;
            let to_idx = line.to;

            // Complex tap ratio t = a * exp(j * phi)
            let a = line.tap;
            let phi = line.shift;
            let t = Complex::from_polar(a, phi);

            // Standard model for transformer with tap on 'from' side
            // Yff = (y_series + y_shunt) / |t|^2
            // Ytt = y_series + y_shunt
            // Yft = -y_series / t*
            // Ytf = -y_series / t
            
            ybus[[from_idx, from_idx]] += (y_series + y_shunt) / (a * a);
            ybus[[to_idx, to_idx]] += y_series + y_shunt;
            ybus[[from_idx, to_idx]] -= y_series / t.conj();
            ybus[[to_idx, from_idx]] -= y_series / t;
        }

        // Bus shunt contributions
        for bus in buses {
            let idx = bus.id as usize;
            ybus[[idx, idx]] += Complex::new(bus.shunt_g, bus.shunt_b);
        }

        ybus
    }

    pub fn print(&self) {
        println!("Power System Summary");
        println!("====================");
        
        println!("\nBuses:");
        println!("{:<5} {:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<10}", "ID", "Name", "V (pu)", "Theta", "P Load", "Q Load", "P Gen", "Q Gen", "Type");
        println!("{}", "-".repeat(90));
        for bus in &self.buses {
            println!(
                "{:<5} {:<15} {:<8.3} {:<8.3} {:<8.3} {:<8.3} {:<8.3} {:<8.3} {:?}",
                bus.id + 1, bus.name, bus.v, bus.theta, bus.p_load, bus.q_load, bus.p_gen, bus.q_gen, bus.bus_type
            );
        }

        println!("\nLines:");
        println!("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}", "From", "To", "R", "X", "B", "Tap", "Status");
        println!("{}", "-".repeat(80));
        for line in &self.lines {
            println!(
                "{:<10} {:<10} {:<10.4} {:<10.4} {:<10.4} {:<10.3} {}",
                line.from + 1, line.to + 1, line.r, line.x, line.b, line.tap, if line.online { "Online" } else { "Offline" }
            );
        }
        println!("\nY-Bus Dimensions: {}x{}", self.ybus.nrows(), self.ybus.ncols());
    }
}
