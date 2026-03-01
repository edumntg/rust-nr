use crate::enums::bustype::BusType;

#[derive(Debug, Clone)]
pub struct Bus {
    pub id: i32,
    pub name: String,
    pub v: f64,
    pub theta: f64,
    pub p_load: f64,
    pub q_load: f64,
    pub p_gen: f64,
    pub q_gen: f64,
    pub shunt_g: f64,
    pub shunt_b: f64,
    pub bus_type: BusType,
}

impl Bus {
    pub fn new(id: i32, name: String, v: f64, theta: f64, p_load: f64, q_load: f64, p_gen: f64, q_gen: f64, shunt_g: f64, shunt_b: f64, bus_type: BusType) -> Self {
        Bus {
            id,
            name,
            v,
            theta,
            p_load,
            q_load,
            p_gen,
            q_gen,
            shunt_g,
            shunt_b,
            bus_type,
        }
    }
}
