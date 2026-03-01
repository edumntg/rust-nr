use std::sync::Arc;
use ndarray::Array2;
use num_complex::{Complex, ComplexFloat};
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

            // Transformer model with tap on 'from' side
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

    pub fn n(&self) -> usize {
        self.buses.len()
    }

    pub fn print_results(&self, x: &Array2<f64>) {
        let n = self.n();
        let mut v = vec![0.0; n];
        let mut theta = vec![0.0; n];

        // Reconstruct the full state from the solution vector
        let mut slack_idx = 0;
        for (i, bus) in self.buses.iter().enumerate() {
            if bus.bus_type == crate::enums::bustype::BusType::Slack {
                slack_idx = i;
            }
            v[i] = bus.v;
            theta[i] = bus.theta;
        }

        let mut var_count = 0;
        for i in 0..n {
            if i != slack_idx {
                theta[i] = x[[var_count, 0]];
                var_count += 1;
            }
        }
        for i in 0..n {
            if self.buses[i].bus_type == crate::enums::bustype::BusType::PQ {
                v[i] = x[[var_count, 0]];
                var_count += 1;
            }
        }

        println!("\nPower Flow Results");
        println!("==================");
        
        println!("\nBus Data:");
        println!("{:<4} {:^8} {:^10} {:^10} {:>12} {:>12} {:>12} {:>12}", 
                 "ID", "Type", "V (pu)", "Ang(rad)", "P Gen", "Q Gen", "P Load", "Q Load");
        println!("{}", "-".repeat(86));

        let g_bus = self.ybus.mapv(|y| y.re);
        let b_bus = self.ybus.mapv(|y| y.im);

        let mut total_p_gen = 0.0;
        let mut total_q_gen = 0.0;
        let mut total_p_load = 0.0;
        let mut total_q_load = 0.0;

        for i in 0..n {
            let mut p_i = 0.0;
            let mut q_i = 0.0;
            for k in 0..n {
                let theta_ik = theta[i] - theta[k];
                p_i += v[i] * v[k] * (g_bus[[i, k]] * theta_ik.cos() + b_bus[[i, k]] * theta_ik.sin());
                q_i += v[i] * v[k] * (g_bus[[i, k]] * theta_ik.sin() - b_bus[[i, k]] * theta_ik.cos());
            }

            let p_gen = p_i + self.buses[i].p_load;
            let q_gen = q_i + self.buses[i].q_load;
            
            total_p_gen += p_gen;
            total_q_gen += q_gen;
            total_p_load += self.buses[i].p_load;
            total_q_load += self.buses[i].q_load;

            let bus_type_str = format!("{:?}", self.buses[i].bus_type);
            println!(
                "{:<4} {:^8} {:^10.4} {:^10.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                self.buses[i].id + 1, bus_type_str, v[i], theta[i], p_gen, q_gen, self.buses[i].p_load, self.buses[i].q_load
            );
        }

        println!("\nLine Flows and Losses:");
        println!("{:<5} {:<5} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}", 
                 "From", "To", "P_ik", "Q_ik", "P_ki", "Q_ki", "P_loss", "Q_loss");
        println!("{}", "-".repeat(90));

        let mut total_p_loss = 0.0;
        let mut total_q_loss = 0.0;

        for line in &self.lines {
            if !line.online { continue; }
            let i = line.from;
            let k = line.to;
            
            let y_series = 1.0 / line.z();
            let y_shunt = Complex::new(0.0, line.b / 2.0);
            let a = line.tap;
            let t = Complex::from_polar(a, line.shift);

            let vi = Complex::from_polar(v[i], theta[i]);
            let vk = Complex::from_polar(v[k], theta[k]);

            let yii = (y_series + y_shunt) / (a * a);
            let ykk = y_series + y_shunt;
            let yik = -y_series / t.conj();
            let yki = -y_series / t;

            let sik = vi * (yii * vi + yik * vk).conj();
            let ski = vk * (yki * vi + ykk * vk).conj();

            let p_ik = sik.re;
            let q_ik = sik.im;
            let p_ki = ski.re;
            let q_ki = ski.im;

            let p_loss = p_ik + p_ki;
            let q_loss = q_ik + q_ki;

            total_p_loss += p_loss;
            total_q_loss += q_loss;

            println!(
                "{:<5} {:<5} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                i + 1, k + 1, p_ik, q_ik, p_ki, q_ki, p_loss, q_loss
            );
        }

        println!("\nSystem Totals (pu):");
        println!("{}", "-".repeat(30));
        println!("{:<15} P: {:>10.4} Q: {:>10.4}", "Total Generation", total_p_gen, total_q_gen);
        println!("{:<15} P: {:>10.4} Q: {:>10.4}", "Total Load", total_p_load, total_q_load);
        println!("{:<15} P: {:>10.4} Q: {:>10.4}", "Total Losses", total_p_loss, total_q_loss);
        println!("{}", "-".repeat(30));
    }

    pub fn initial_x(&self) -> Array2<f64> {
        let n = self.n();
        let mut slack_idx = 0;
        let mut bus_types = Vec::new();
        for (i, bus) in self.buses.iter().enumerate() {
            bus_types.push(bus.bus_type);
            if bus.bus_type == crate::enums::bustype::BusType::Slack {
                slack_idx = i;
            }
        }

        let mut var_count = 0;
        for i in 0..n {
            if i != slack_idx {
                var_count += 1;
            }
        }
        for i in 0..n {
            if bus_types[i] == crate::enums::bustype::BusType::PQ {
                var_count += 1;
            }
        }

        let mut x = Array2::<f64>::zeros((var_count, 1));
        let mut current_var = 0;

        for i in 0..n {
            if i != slack_idx {
                x[[current_var, 0]] = self.buses[i].theta;
                current_var += 1;
            }
        }
        for i in 0..n {
            if bus_types[i] == crate::enums::bustype::BusType::PQ {
                x[[current_var, 0]] = self.buses[i].v;
                current_var += 1;
            }
        }

        x
    }

    pub fn prepare(&self) -> (Box<dyn Fn(&Array2<f64>) -> Array2<f64>>, Box<dyn Fn(&Array2<f64>) -> Array2<f64>>) {
        let n = self.buses.len();
        let g_bus = Arc::new(self.ybus.mapv(|y| y.re));
        let b_bus = Arc::new(self.ybus.mapv(|y| y.im));

        let mut slack_idx = 0;
        let mut bus_types = Vec::new();
        let mut p_spec = Vec::new();
        let mut q_spec = Vec::new();
        let mut v_init = Vec::new();
        let mut theta_init = Vec::new();

        for (i, bus) in self.buses.iter().enumerate() {
            bus_types.push(bus.bus_type);
            if bus.bus_type == crate::enums::bustype::BusType::Slack {
                slack_idx = i;
            }
            p_spec.push(bus.p_gen - bus.p_load);
            q_spec.push(bus.q_gen - bus.q_load);
            v_init.push(bus.v);
            theta_init.push(bus.theta);
        }

        let mut theta_map = vec![None; n];
        let mut v_map = vec![None; n];
        let mut var_count = 0;

        for i in 0..n {
            if i != slack_idx {
                theta_map[i] = Some(var_count);
                var_count += 1;
            }
        }
        for i in 0..n {
            if bus_types[i] == crate::enums::bustype::BusType::PQ {
                v_map[i] = Some(var_count);
                var_count += 1;
            }
        }

        let bus_types = Arc::new(bus_types);
        let p_spec = Arc::new(p_spec);
        let q_spec = Arc::new(q_spec);
        let v_init = Arc::new(v_init);
        let theta_init = Arc::new(theta_init);
        let theta_map = Arc::new(theta_map);
        let v_map = Arc::new(v_map);

        let f = {
            let g_bus = Arc::clone(&g_bus);
            let b_bus = Arc::clone(&b_bus);
            let p_spec = Arc::clone(&p_spec);
            let q_spec = Arc::clone(&q_spec);
            let v_init = Arc::clone(&v_init);
            let theta_init = Arc::clone(&theta_init);
            let bus_types = Arc::clone(&bus_types);
            let theta_map = Arc::clone(&theta_map);
            let v_map = Arc::clone(&v_map);
            
            move |x: &Array2<f64>| {
                let mut fx = Array2::<f64>::zeros((var_count, 1));
                let mut v = vec![0.0; n];
                let mut theta = vec![0.0; n];
                let mut cos_t = vec![0.0; n];
                let mut sin_t = vec![0.0; n];

                for i in 0..n {
                    v[i] = v_map[i].map_or(v_init[i], |idx| x[[idx, 0]]);
                    theta[i] = theta_map[i].map_or(theta_init[i], |idx| x[[idx, 0]]);
                    cos_t[i] = theta[i].cos();
                    sin_t[i] = theta[i].sin();
                }

                for i in 0..n {
                    if i == slack_idx { continue; }

                    let mut p_i = 0.0;
                    for k in 0..n {
                        // Use trig identities to avoid expensive cos(theta_i - theta_k)
                        let cos_ik = cos_t[i] * cos_t[k] + sin_t[i] * sin_t[k];
                        let sin_ik = sin_t[i] * cos_t[k] - cos_t[i] * sin_t[k];
                        p_i += v[i] * v[k] * (g_bus[[i, k]] * cos_ik + b_bus[[i, k]] * sin_ik);
                    }
                    if let Some(idx) = theta_map[i] {
                        fx[[idx, 0]] = p_i - p_spec[i];
                    }

                    if bus_types[i] == crate::enums::bustype::BusType::PQ {
                        let mut q_i = 0.0;
                        for k in 0..n {
                            let cos_ik = cos_t[i] * cos_t[k] + sin_t[i] * sin_t[k];
                            let sin_ik = sin_t[i] * cos_t[k] - cos_t[i] * sin_t[k];
                            q_i += v[i] * v[k] * (g_bus[[i, k]] * sin_ik - b_bus[[i, k]] * cos_ik);
                        }
                        if let Some(idx) = v_map[i] {
                            fx[[idx, 0]] = q_i - q_spec[i];
                        }
                    }
                }
                fx
            }
        };

        let df = {
            let g_bus = Arc::clone(&g_bus);
            let b_bus = Arc::clone(&b_bus);
            let v_init = Arc::clone(&v_init);
            let theta_init = Arc::clone(&theta_init);
            let theta_map = Arc::clone(&theta_map);
            let v_map = Arc::clone(&v_map);

            move |x: &Array2<f64>| {
                let mut dfx = Array2::<f64>::zeros((var_count, var_count));
                let mut v = vec![0.0; n];
                let mut theta = vec![0.0; n];
                let mut cos_t = vec![0.0; n];
                let mut sin_t = vec![0.0; n];

                for i in 0..n {
                    v[i] = v_map[i].map_or(v_init[i], |idx| x[[idx, 0]]);
                    theta[i] = theta_map[i].map_or(theta_init[i], |idx| x[[idx, 0]]);
                    cos_t[i] = theta[i].cos();
                    sin_t[i] = theta[i].sin();
                }

                // Pre-calculate P and Q to reuse in diagonal Jacobian elements
                let mut p = vec![0.0; n];
                let mut q = vec![0.0; n];
                for i in 0..n {
                    for k in 0..n {
                        let cos_ik = cos_t[i] * cos_t[k] + sin_t[i] * sin_t[k];
                        let sin_ik = sin_t[i] * cos_t[k] - cos_t[i] * sin_t[k];
                        p[i] += v[i] * v[k] * (g_bus[[i, k]] * cos_ik + b_bus[[i, k]] * sin_ik);
                        q[i] += v[i] * v[k] * (g_bus[[i, k]] * sin_ik - b_bus[[i, k]] * cos_ik);
                    }
                }

                for i in 0..n {
                    let row_p = theta_map[i];
                    let row_q = v_map[i];

                    for k in 0..n {
                        let col_theta = theta_map[k];
                        let col_v = v_map[k];
                        let cos_ik = cos_t[i] * cos_t[k] + sin_t[i] * sin_t[k];
                        let sin_ik = sin_t[i] * cos_t[k] - cos_t[i] * sin_t[k];

                        if let (Some(rp), Some(ct)) = (row_p, col_theta) {
                            dfx[[rp, ct]] = if i == k {
                                -q[i] - v[i] * v[i] * b_bus[[i, i]]
                            } else {
                                v[i] * v[k] * (g_bus[[i, k]] * sin_ik - b_bus[[i, k]] * cos_ik)
                            };
                        }

                        if let (Some(rp), Some(cv)) = (row_p, col_v) {
                            dfx[[rp, cv]] = if i == k {
                                p[i] / v[i] + v[i] * g_bus[[i, i]]
                            } else {
                                v[i] * (g_bus[[i, k]] * cos_ik + b_bus[[i, k]] * sin_ik)
                            };
                        }

                        if let (Some(rq), Some(ct)) = (row_q, col_theta) {
                            dfx[[rq, ct]] = if i == k {
                                p[i] - v[i] * v[i] * g_bus[[i, i]]
                            } else {
                                -v[i] * v[k] * (g_bus[[i, k]] * cos_ik + b_bus[[i, k]] * sin_ik)
                            };
                        }

                        if let (Some(rq), Some(cv)) = (row_q, col_v) {
                            dfx[[rq, cv]] = if i == k {
                                q[i] / v[i] - v[i] * b_bus[[i, i]]
                            } else {
                                v[i] * (g_bus[[i, k]] * sin_ik - b_bus[[i, k]] * cos_ik)
                            };
                        }
                    }
                }
                dfx
            }
        };

        (Box::new(f), Box::new(df))
    }
}
