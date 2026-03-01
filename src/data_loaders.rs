use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::models::bus::Bus;
use crate::models::line::Line;
use crate::models::power_system::PowerSystem;
use crate::enums::bustype::BusType;

// A simple state machine enum to track which section we are parsing
#[derive(PartialEq)]
enum ParseState {
    Searching,
    ReadingBuses,
    ReadingBranches,
}

pub fn load_ieee(filename: &str) -> PowerSystem {
    let mut buses: Vec<Bus> = Vec::new();
    let mut lines: Vec<Line> = Vec::new();

    let file = File::open(filename).expect("Unable to open file");
    let reader = BufReader::new(file);

    let mut state = ParseState::Searching;

    for line_result in reader.lines() {
        let line = line_result.expect("Unable to read line");

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // 1. Check for Section Headers
        if line.contains("BUS DATA FOLLOWS") {
            state = ParseState::ReadingBuses;
            continue;
        } else if line.contains("BRANCH DATA FOLLOWS") {
            state = ParseState::ReadingBranches;
            continue;
        }

        // 2. Check for the End of Section Flag
        if line.trim_start().starts_with("-999") {
            state = ParseState::Searching;
            continue;
        }

        // 3. Parse Data based on the Current State using Fixed-Width Slicing
        match state {
            ParseState::ReadingBuses => {
                let bus_num_str = get_slice(&line, 0, 4);
                let name = get_slice(&line, 5, 17);
                let bus_type_str = get_slice(&line, 24, 26);
                let pl_str = get_slice(&line, 40, 49);
                let ql_str = get_slice(&line, 49, 59);
                let pgen_str = get_slice(&line, 59, 69);
                let qgen_str = get_slice(&line, 69, 79);

                let bus_type = match bus_type_str.parse::<i32>().unwrap_or(0) {
                    0 => BusType::PQ,
                    2 => BusType::PV,
                    3 => BusType::Slack,
                    _ => panic!("Unknown bus type code: {}", bus_type_str),
                };

                let bus = Bus::new(
                    bus_num_str.parse().unwrap(),
                    name.to_string(),
                    1.0, // v
                    0.0, // theta
                    pl_str.parse().unwrap_or(0.0),
                    ql_str.parse().unwrap_or(0.0),
                    pgen_str.parse().unwrap_or(0.0), // p_gen
                    qgen_str.parse().unwrap_or(0.0), // q_gen
                    0.0, // shunt_g
                    0.0, // shunt_b
                    bus_type,
                );
                buses.push(bus);
            }
            ParseState::ReadingBranches => {
                let tap_bus_str = get_slice(&line, 0, 4);
                let z_bus_str = get_slice(&line, 5, 9);
                let r_str = get_slice(&line, 19, 29);
                let x_str = get_slice(&line, 29, 40);
                let b_str = get_slice(&line, 40, 50);

                let line_struct = Line::new(
                    tap_bus_str.parse().unwrap(),
                    z_bus_str.parse().unwrap(),
                    r_str.parse().unwrap(),
                    x_str.parse().unwrap(),
                    b_str.parse().unwrap(),
                    1.0,  // tap
                    0.0,  // shift
                    true, // online
                );
                lines.push(line_struct);
            }
            ParseState::Searching => {
                continue;
            }
        }
    }

    PowerSystem::new(buses, lines)
}

fn get_slice(s: &str, start: usize, end: usize) -> &str {
    if s.len() >= end {
        s[start..end].trim()
    } else if s.len() > start {
        s[start..].trim()
    } else {
        ""
    }
}
