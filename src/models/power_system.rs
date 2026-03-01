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

    pub fn n(&self) -> usize {
        self.buses.len()
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

        // Theta initial values
        for i in 0..n {
            if i != slack_idx {
                x[[current_var, 0]] = self.buses[i].theta;
                current_var += 1;
            }
        }
        // V initial values
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

                for i in 0..n {
                    v[i] = v_map[i].map_or(v_init[i], |idx| x[[idx, 0]]);
                    theta[i] = theta_map[i].map_or(theta_init[i], |idx| x[[idx, 0]]);
                }

                for i in 0..n {
                    if i == slack_idx { continue; }

                    let mut p_i = 0.0;
                    for k in 0..n {
                        let theta_ik = theta[i] - theta[k];
                        p_i += v[i] * v[k] * (g_bus[[i, k]] * theta_ik.cos() + b_bus[[i, k]] * theta_ik.sin());
                    }
                    if let Some(idx) = theta_map[i] {
                        fx[[idx, 0]] = p_i - p_spec[i];
                    }

                    if bus_types[i] == crate::enums::bustype::BusType::PQ {
                        let mut q_i = 0.0;
                        for k in 0..n {
                            let theta_ik = theta[i] - theta[k];
                            q_i += v[i] * v[k] * (g_bus[[i, k]] * theta_ik.sin() - b_bus[[i, k]] * theta_ik.cos());
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

                for i in 0..n {
                    v[i] = v_map[i].map_or(v_init[i], |idx| x[[idx, 0]]);
                    theta[i] = theta_map[i].map_or(theta_init[i], |idx| x[[idx, 0]]);
                }

                let mut p = vec![0.0; n];
                let mut q = vec![0.0; n];
                for i in 0..n {
                    for k in 0..n {
                        let theta_ik = theta[i] - theta[k];
                        let cos_ik = theta_ik.cos();
                        let sin_ik = theta_ik.sin();
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
                        let theta_ik = theta[i] - theta[k];
                        let cos_ik = theta_ik.cos();
                        let sin_ik = theta_ik.sin();

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
