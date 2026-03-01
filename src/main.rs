mod models;
mod enums;
mod loaders;

use crate::loaders::data_loaders::load_ieee;
use crate::models::power_system::PowerSystem;

fn main() {
    // Load IEEE-14 bus system
    let power_system: PowerSystem = load_ieee("src/datasets/ieee14.txt");
    power_system.print()
}
