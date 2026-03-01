use std::fs::File;
use std::io::{BufRead, BufReader};
use std::f64::consts::PI;

use crate::models::bus::Bus;
use crate::models::line::Line;
use crate::models::power_system::PowerSystem;
use crate::enums::bustype::BusType;

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
    let mut reader = BufReader::new(file);

    let mut first_line = String::new();
    reader.read_line(&mut first_line).expect("Unable to read first line");
    let base_mva: f64 = get_slice(&first_line, 31, 37).parse().unwrap_or(100.0);

    let mut state = ParseState::Searching;

    for line_result in reader.lines() {
        let line = line_result.expect("Unable to read line");

        if line.trim().is_empty() {
            continue;
        }

        if line.contains("BUS DATA FOLLOWS") {
            state = ParseState::ReadingBuses;
            continue;
        } else if line.contains("BRANCH DATA FOLLOWS") {
            state = ParseState::ReadingBranches;
            continue;
        }

        if line.trim_start().starts_with("-999") || line.trim_start().starts_with("-99") {
            state = ParseState::Searching;
            continue;
        }

        match state {
            ParseState::ReadingBuses => {
                let bus_id: i32 = get_slice(&line, 0, 4).parse().expect("Failed to parse bus ID");
                let name = get_slice(&line, 5, 17);
                let bus_type_code: i32 = get_slice(&line, 24, 26).parse().unwrap_or(0);
                let v_pu: f64 = get_slice(&line, 27, 33).parse().unwrap_or(1.0);
                let theta_deg: f64 = get_slice(&line, 33, 40).parse().unwrap_or(0.0);
                let p_load_mw: f64 = get_slice(&line, 40, 49).parse().unwrap_or(0.0);
                let q_load_mvar: f64 = get_slice(&line, 49, 59).parse().unwrap_or(0.0);
                let p_gen_mw: f64 = get_slice(&line, 60, 69).parse().unwrap_or(0.0);
                let q_gen_mvar: f64 = get_slice(&line, 70, 79).parse().unwrap_or(0.0);
                let shunt_g: f64 = get_slice(&line, 108, 116).parse().unwrap_or(0.0);
                let shunt_b: f64 = get_slice(&line, 116, 124).parse().unwrap_or(0.0);

                let bus_type = match bus_type_code {
                    0 | 1 => BusType::PQ,
                    2 => BusType::PV,
                    3 => BusType::Slack,
                    _ => BusType::PQ,
                };

                let bus = Bus::new(
                    bus_id - 1, // Shift to 0-based
                    name.to_string(),
                    v_pu,
                    theta_deg * PI / 180.0,
                    p_load_mw / base_mva,
                    q_load_mvar / base_mva,
                    p_gen_mw / base_mva,
                    q_gen_mvar / base_mva,
                    shunt_g,
                    shunt_b,
                    bus_type,
                );
                buses.push(bus);
            }
            ParseState::ReadingBranches => {
                let from_id: i32 = get_slice(&line, 0, 4).parse().expect("Failed to parse from_id");
                let to_id: i32 = get_slice(&line, 5, 9).parse().expect("Failed to parse to_id");
                let r: f64 = get_slice(&line, 19, 29).parse().unwrap_or(0.0);
                let x: f64 = get_slice(&line, 29, 40).parse().unwrap_or(0.0);
                let b: f64 = get_slice(&line, 40, 50).parse().unwrap_or(0.0);
                let tap: f64 = get_slice(&line, 76, 82).parse().unwrap_or(0.0);
                let shift: f64 = get_slice(&line, 83, 90).parse().unwrap_or(0.0);

                let line_struct = Line::new(
                    from_id - 1, // Shift to 0-based
                    to_id - 1,   // Shift to 0-based
                    r,
                    x,
                    b,
                    if tap == 0.0 { 1.0 } else { tap },
                    shift * PI / 180.0,
                    true,
                );
                lines.push(line_struct);
            }
            ParseState::Searching => {}
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
