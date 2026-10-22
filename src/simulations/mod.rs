mod argon;
mod blackhole;
mod chain_reaction;
mod double_pendulum;
mod galaxies;
mod ising;
mod light;
mod navier_stokes;
mod planets;
mod slope;
mod wave_packet;
mod top;
mod vortex_street;
mod water;

use crate::simulation::Simulation;

/// A registered simulation.
pub struct Entry {
    pub name: &'static str,
    pub create: fn() -> Box<dyn Simulation>,
}

/// All simulations. A simulation's number is its position here, starting at 1,
/// so append new ones at the end to keep existing numbers stable.
pub const SIMULATIONS: &[Entry] = &[
    Entry { name: "slope", create: || Box::new(slope::Slope::new()) },
    Entry { name: "planets", create: || Box::new(planets::Planets::new()) },
    Entry { name: "double-pendulum", create: || Box::new(double_pendulum::DoublePendulum::new()) },
    Entry { name: "blackhole", create: || Box::new(blackhole::BlackHole::new()) },
    Entry { name: "ising", create: || Box::new(ising::Ising::new()) },
    Entry { name: "chain-reaction", create: || Box::new(chain_reaction::ChainReaction::new()) },
    Entry { name: "top", create: || Box::new(top::Top::new()) },
    Entry { name: "galaxies", create: || Box::new(galaxies::Galaxies::new()) },
    Entry { name: "argon", create: || Box::new(argon::Argon::new()) },
    Entry { name: "wavepacket", create: || Box::new(wave_packet::WavePacket::new()) },
    Entry { name: "light", create: || Box::new(light::Light::new()) },
    Entry { name: "vortex-street", create: || Box::new(vortex_street::VortexStreet::new()) },
    Entry { name: "navier-stokes", create: || Box::new(navier_stokes::NavierStokes::new()) },
    Entry { name: "water", create: || Box::new(water::Water::new()) },
];

/// Finds a simulation by name (case-insensitive) or by number.
/// Returns its number too.
pub fn find(query: &str) -> Option<(usize, &'static Entry)> {
    let index = match query.parse::<usize>() {
        Ok(number) => number.checked_sub(1)?,
        Err(_) => SIMULATIONS
            .iter()
            .position(|entry| entry.name.eq_ignore_ascii_case(query))?,
    };
    SIMULATIONS.get(index).map(|entry| (index + 1, entry))
}

/// One "number  name" line per simulation, for usage messages.
pub fn list() -> String {
    SIMULATIONS
        .iter()
        .enumerate()
        .map(|(i, entry)| format!("  {}  {}", i + 1, entry.name))
        .collect::<Vec<_>>()
        .join("\n")
}
