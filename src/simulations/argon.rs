use macroquad::prelude::*;
use macroquad::rand::RandGenerator;

use crate::simulation::Simulation;

// Units: reduced. Length sigma, energy epsilon, mass m, k_B = 1; time is then
// tau = sigma sqrt(m / epsilon). The HUD converts to real units with the constants below.
// Argon (Michels, Wijker & Wijker 1949, the set Verlet used in 1967): epsilon / k_B = 119.8 K,
// sigma = 3.405 A, m = 39.948 u.
const EPSILON_KELVIN: f64 = 119.8;
const SIGMA_ANGSTROM: f64 = 3.405;
#[cfg_attr(not(test), allow(dead_code))]
const MASS_U: f64 = 39.948;
/// tau in picoseconds (a test recomputes it from epsilon, sigma and m).
const TAU_PS: f64 = 2.156_349;
/// 2D pressure unit epsilon / sigma^2 in mN/m (force per length, not Pa).
const PRESSURE_MN_PER_M: f64 = 14.266_12;

/// Lennard-Jones potential, truncated at this distance and shifted to be continuous there.
const CUTOFF: f64 = 3.5;
const U_CUTOFF: f64 = {
    let r2 = CUTOFF * CUTOFF;
    let r6 = r2 * r2 * r2;
    4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
};
/// Atoms closer than this count as neighbours (between the crystal's first two shells).
const NEIGHBOUR_DISTANCE: f64 = 1.5;
/// Position of the potential minimum, 2^(1/6).
#[cfg_attr(not(test), allow(dead_code))]
const R_MIN: f64 = 1.122_462_048_309_373;

/// Time step: 10.78 fs. The fastest vibration of the crystal has omega = 18.5 / tau, so 68 steps
/// per period.
const DT: f64 = 0.005;
/// Friction of the heat bath, 1/tau.
const FRICTION: f64 = 0.2;

/// Zero-temperature spacing of the triangular crystal for this potential.
const LATTICE_SPACING: f64 = 1.111_940_36;
const BOX: DVec2 = DVec2::new(72.0, 30.0);
/// The starting crystal island: columns and rows of atoms.
const ISLAND: (usize, usize) = (34, 22);
const START_KELVIN: f64 = 30.0;
const KELVIN_STEP: f64 = 2.0;
const MIN_KELVIN: f64 = 2.0;
const MAX_KELVIN: f64 = 240.0;
const START_SPEED: f64 = 3.0; // tau per second of real time
const MIN_SPEED: f64 = 0.75;
const MAX_SPEED: f64 = 12.0;
const SEED: u64 = 12;
/// Time constants (tau) of the moving averages on the HUD.
const TEMPERATURE_AVERAGE_TIME: f64 = 2.0;
const PRESSURE_AVERAGE_TIME: f64 = 20.0;

const ATOM_RADIUS: f32 = 0.5; // sigma

/// Energy and r times the force of one pair at squared distance `r2` (zero beyond the cutoff):
///   u(r) = 4 [r^-12 - r^-6] - u_c,   r F(r) = 24 [2 r^-12 - r^-6].
/// The force on an atom is `r F / r2` times the vector from the other atom. No square root needed.
fn pair(r2: f64) -> (f64, f64) {
    if r2 >= CUTOFF * CUTOFF {
        return (0.0, 0.0);
    }
    let s6 = 1.0 / (r2 * r2 * r2);
    let s12 = s6 * s6;
    (4.0 * (s12 - s6) - U_CUTOFF, 24.0 * (2.0 * s12 - s6))
}

/// Atoms of a triangular lattice: row `j` at y = j a sqrt(3)/2, odd rows shifted by a/2.
fn triangular_lattice(columns: usize, rows: usize, spacing: f64, origin: DVec2) -> Vec<DVec2> {
    let mut atoms = Vec::with_capacity(columns * rows);
    for j in 0..rows {
        for i in 0..columns {
            let shift = if j % 2 == 1 { 0.5 } else { 0.0 };
            atoms.push(origin + dvec2((i as f64 + shift) * spacing, j as f64 * spacing * 3f64.sqrt() / 2.0));
        }
    }
    atoms
}

/// Atoms interacting through the Lennard-Jones potential in a periodic 2D box (molecular
/// dynamics), with a Langevin heat bath: a monolayer of argon, which starts as a cold crystal
/// island. UP/DOWN change the bath temperature; the crystal melts into a liquid strip beside
/// vapour near 47 K, the liquid/vapour boundary disappears near 60 K (the 2D critical point), and
/// cooling brings droplets and a polycrystal back. T switches the bath off, giving Newton's
/// equations. BAOAB splitting of the Langevin equation (velocity Verlet without the bath),
/// cell list for neighbours.
/// Not modelled: the third dimension (these temperatures are about half the bulk ones), the
/// substrate, the true Ar-Ar interaction (three-body forces), the potential's tail beyond 3.5
/// sigma, quantum effects, and finite size (748 atoms).
pub struct Argon {
    /// sigma, inside the box.
    pos: Vec<DVec2>,
    /// sigma / tau.
    v: Vec<DVec2>,
    /// epsilon / sigma, also the acceleration (m = 1).
    force: Vec<DVec2>,
    /// Neighbours within 1.5 sigma, for colouring.
    coordination: Vec<u8>,
    box_size: DVec2,
    /// Atom indices per cell of the neighbour grid, `cells_x * cells_y` cells of side >= cutoff.
    cells: Vec<Vec<u32>>,
    cells_x: usize,
    cells_y: usize,
    /// Set by `update_forces`: sum of u over pairs, and sum of r F over pairs.
    potential_energy: f64,
    virial: f64,
    /// epsilon / k_B.
    bath_temperature: f64,
    thermostat_on: bool,
    rng: RandGenerator,
    /// Total energy when the thermostat was last switched off.
    energy_reference: f64,
    temperature_average: f64,
    pressure_average: f64,
    /// tau per second of real time.
    speed: f64,
    /// tau.
    t: f64,
}

impl Argon {
    pub fn new() -> Self {
        let (columns, rows) = ISLAND;
        // Centre the island in the box.
        let size = dvec2((columns as f64 - 0.5) * LATTICE_SPACING, (rows - 1) as f64 * LATTICE_SPACING * 3f64.sqrt() / 2.0);
        let pos = triangular_lattice(columns, rows, LATTICE_SPACING, (BOX - size) / 2.0);

        // Normal velocities with no net motion, scaled so K/N is exactly the starting temperature.
        let rng = RandGenerator::new();
        rng.srand(SEED);
        let mut v: Vec<DVec2> = pos.iter().map(|_| normal_pair(&rng)).collect();
        let mean = v.iter().copied().sum::<DVec2>() / v.len() as f64;
        let k: f64 = v.iter().map(|v| 0.5 * (*v - mean).length_squared()).sum();
        let scale = (START_KELVIN / EPSILON_KELVIN / (k / v.len() as f64)).sqrt();
        for v in &mut v {
            *v = (*v - mean) * scale;
        }
        Self::with_atoms(BOX, pos, v, START_KELVIN / EPSILON_KELVIN, true)
    }

    /// Atoms at `pos` with velocities `v` in a periodic box.
    fn with_atoms(box_size: DVec2, pos: Vec<DVec2>, v: Vec<DVec2>, bath_temperature: f64, thermostat_on: bool) -> Self {
        let cells_x = (box_size.x / CUTOFF).floor() as usize;
        let cells_y = (box_size.y / CUTOFF).floor() as usize;
        // With fewer than 3 cells a side, wrapped neighbour cells coincide and pairs count twice.
        assert!(cells_x >= 3 && cells_y >= 3, "box too small for the cell list");
        let rng = RandGenerator::new();
        rng.srand(SEED);
        let n = pos.len();
        let mut argon = Self {
            force: vec![DVec2::ZERO; n],
            coordination: vec![0; n],
            pos,
            v,
            box_size,
            cells: vec![Vec::new(); cells_x * cells_y],
            cells_x,
            cells_y,
            potential_energy: 0.0,
            virial: 0.0,
            bath_temperature,
            thermostat_on,
            rng,
            energy_reference: 0.0,
            temperature_average: 0.0,
            pressure_average: 0.0,
            speed: START_SPEED,
            t: 0.0,
        };
        argon.update_forces();
        argon.energy_reference = argon.energy();
        argon.temperature_average = argon.temperature();
        argon.pressure_average = argon.pressure();
        argon
    }

    /// Separation from `b` to `a` using the nearest periodic image.
    fn minimum_image(&self, mut d: DVec2) -> DVec2 {
        d.x -= self.box_size.x * (d.x / self.box_size.x).round();
        d.y -= self.box_size.y * (d.y / self.box_size.y).round();
        d
    }

    /// Fills `force`, `coordination`, `potential_energy` and `virial` using a cell list: every
    /// pair is visited once, inside a cell and with 4 of the 8 neighbouring cells.
    fn update_forces(&mut self) {
        let (nx, ny) = (self.cells_x, self.cells_y);
        for cell in &mut self.cells {
            cell.clear();
        }
        for (i, p) in self.pos.iter().enumerate() {
            let cx = ((p.x / self.box_size.x * nx as f64) as usize).min(nx - 1);
            let cy = ((p.y / self.box_size.y * ny as f64) as usize).min(ny - 1);
            self.cells[cy * nx + cx].push(i as u32);
        }
        self.force.fill(DVec2::ZERO);
        self.coordination.fill(0);
        let (mut energy, mut virial) = (0.0, 0.0);

        for cy in 0..ny {
            for cx in 0..nx {
                let here = cy * nx + cx;
                // East, north-east, north and north-west neighbours, wrapping around the box.
                let others = [
                    cy * nx + (cx + 1) % nx,
                    ((cy + 1) % ny) * nx + (cx + 1) % nx,
                    ((cy + 1) % ny) * nx + cx,
                    ((cy + 1) % ny) * nx + (cx + nx - 1) % nx,
                ];
                for a in 0..self.cells[here].len() {
                    let i = self.cells[here][a] as usize;
                    // Pairs inside the cell, then with the neighbouring cells.
                    let inside = self.cells[here][a + 1..].iter();
                    let outside = others.iter().flat_map(|&other| self.cells[other].iter());
                    for &j in inside.chain(outside) {
                        let j = j as usize;
                        let d = self.minimum_image(self.pos[i] - self.pos[j]);
                        let r2 = d.length_squared();
                        if r2 < NEIGHBOUR_DISTANCE * NEIGHBOUR_DISTANCE {
                            self.coordination[i] += 1;
                            self.coordination[j] += 1;
                        }
                        let (u, rf) = pair(r2);
                        if rf != 0.0 {
                            let f = d * (rf / r2);
                            self.force[i] += f;
                            self.force[j] -= f;
                            energy += u;
                            virial += rf;
                        }
                    }
                }
            }
        }
        self.potential_energy = energy;
        self.virial = virial;
    }

    fn kinetic_energy(&self) -> f64 {
        self.v.iter().map(|v| 0.5 * v.length_squared()).sum()
    }

    /// From equipartition, 2 degrees of freedom per atom: T = K / N.
    fn temperature(&self) -> f64 {
        self.kinetic_energy() / self.pos.len() as f64
    }

    fn area(&self) -> f64 {
        self.box_size.x * self.box_size.y
    }

    /// From the virial theorem: P = (N T + W / 2) / A, with W the sum of r F over pairs.
    fn pressure(&self) -> f64 {
        (self.kinetic_energy() + 0.5 * self.virial) / self.area()
    }

    fn energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy
    }

    #[cfg(test)]
    fn momentum(&self) -> DVec2 {
        self.v.iter().copied().sum()
    }

    fn wrap(&mut self, i: usize) {
        self.pos[i].x = self.pos[i].x.rem_euclid(self.box_size.x);
        self.pos[i].y = self.pos[i].y.rem_euclid(self.box_size.y);
    }
}

/// A pair of independent standard normal numbers (Box-Muller).
fn normal_pair(rng: &RandGenerator) -> DVec2 {
    let s = (-2.0 * (1.0 - rng.gen_range(0.0_f64, 1.0)).ln()).sqrt();
    let angle = std::f64::consts::TAU * rng.gen_range(0.0_f64, 1.0);
    dvec2(s * angle.cos(), s * angle.sin())
}

impl Simulation for Argon {
    /// One BAOAB step: half kick (B), half drift (A), the bath's friction and noise solved
    /// exactly (O), half drift, new forces, half kick. Without the bath it is velocity Verlet:
    /// second order, time-reversible, symplectic, so the energy error oscillates.
    fn step(&mut self, dt: f64) {
        let half = dt / 2.0;
        for i in 0..self.pos.len() {
            self.v[i] += self.force[i] * half;
            self.pos[i] += self.v[i] * half;
        }
        if self.thermostat_on {
            let c = (-FRICTION * dt).exp();
            let noise = ((1.0 - c * c) * self.bath_temperature).sqrt();
            for v in &mut self.v {
                *v = *v * c + normal_pair(&self.rng) * noise;
            }
        }
        for i in 0..self.pos.len() {
            self.pos[i] += self.v[i] * half;
            self.wrap(i);
        }
        self.update_forces();
        for i in 0..self.pos.len() {
            self.v[i] += self.force[i] * half;
        }

        self.t += dt;
        self.temperature_average += (self.temperature() - self.temperature_average) * (dt / TEMPERATURE_AVERAGE_TIME).min(1.0);
        self.pressure_average += (self.pressure() - self.pressure_average) * (dt / PRESSURE_AVERAGE_TIME).min(1.0);
    }

    fn is_over(&self) -> bool {
        false
    }

    fn dt(&self) -> f64 {
        DT
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn world_bounds(&self) -> Rect {
        // 3:2 like the default window; room above the box for the HUD.
        Rect::new(-2.57, -4.29, 77.14, 51.43)
    }

    fn draw(&self) {
        draw_rectangle_lines(0.0, 0.0, self.box_size.x as f32, self.box_size.y as f32, 0.15, GRAY);
        for (p, &neighbours) in self.pos.iter().zip(&self.coordination) {
            let color = match neighbours {
                0..=1 => SKYBLUE,
                2..=4 => ORANGE,
                5 => GOLD,
                6 => GREEN,
                _ => RED,
            };
            draw_poly(p.x as f32, p.y as f32, 12, ATOM_RADIUS, 0.0, color);
        }
    }

    fn handle_input(&mut self) {
        let step = KELVIN_STEP / EPSILON_KELVIN;
        let (min, max) = (MIN_KELVIN / EPSILON_KELVIN, MAX_KELVIN / EPSILON_KELVIN);
        if is_key_pressed(KeyCode::Up) {
            self.bath_temperature = (self.bath_temperature + step).min(max);
        }
        if is_key_pressed(KeyCode::Down) {
            self.bath_temperature = (self.bath_temperature - step).max(min);
        }
        if is_key_pressed(KeyCode::T) {
            self.thermostat_on = !self.thermostat_on;
            if !self.thermostat_on {
                self.energy_reference = self.energy();
            }
        }
        if is_key_pressed(KeyCode::Right) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Left) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
        if is_key_pressed(KeyCode::R) {
            *self = Self::new();
        }
    }

    fn hud(&self) -> Vec<String> {
        let n = self.pos.len();
        let ideal = n as f64 * self.temperature() / self.area() * PRESSURE_MN_PER_M;
        let last = if self.thermostat_on {
            format!("speed {:.1} ps/s (LEFT/RIGHT)   R restarts", self.speed * TAU_PS)
        } else {
            format!("energy drift = {:+.1e} eps/atom   R restarts", (self.energy() - self.energy_reference) / n as f64)
        };
        vec![
            format!(
                "t = {:.1} ps   {} argon atoms (2D)   box {:.0} x {:.0} A",
                self.t * TAU_PS,
                n,
                self.box_size.x * SIGMA_ANGSTROM,
                self.box_size.y * SIGMA_ANGSTROM
            ),
            format!(
                "T = {:.1} K   bath {:.0} K (UP/DOWN)   thermostat {} (T)",
                self.temperature_average * EPSILON_KELVIN,
                self.bath_temperature * EPSILON_KELVIN,
                if self.thermostat_on { "ON" } else { "OFF" }
            ),
            format!("P = {:.1} mN/m   ideal gas: {:.1} mN/m", self.pressure_average * PRESSURE_MN_PER_M, ideal),
            format!("T* = {:.2}   2D: melts ~47 K, critical ~60 K", self.temperature_average),
            last,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    fn run(argon: &mut Argon, steps: usize) {
        for _ in 0..steps {
            argon.step(DT);
        }
    }

    /// 400 atoms on a 20 x 20 square lattice at density 0.7, with normal velocities at `t`
    /// (no net motion), thermostat as given.
    fn lattice_fluid(speed_of: impl Fn(&RandGenerator) -> DVec2, thermostat_on: bool, bath: f64, seed: u64) -> Argon {
        let side = (400.0f64 / 0.7).sqrt();
        let spacing = side / 20.0;
        let pos: Vec<DVec2> = (0..400).map(|k| dvec2((k % 20) as f64 + 0.5, (k / 20) as f64 + 0.5) * spacing).collect();
        let rng = RandGenerator::new();
        rng.srand(seed);
        let mut v: Vec<DVec2> = pos.iter().map(|_| speed_of(&rng)).collect();
        let mean = v.iter().copied().sum::<DVec2>() / 400.0;
        for v in &mut v {
            *v -= mean;
        }
        Argon::with_atoms(dvec2(side, side), pos, v, bath, thermostat_on)
    }

    /// The "fluid" of the plan: settled with the thermostat at T = 0.7 for 4000 steps.
    fn fluid() -> Argon {
        let mut argon = lattice_fluid(|rng| normal_pair(rng) * 0.7f64.sqrt(), true, 0.7, 1);
        run(&mut argon, 4000);
        argon
    }

    /// A copy of the state with the thermostat off.
    fn isolated_copy(argon: &Argon) -> Argon {
        Argon::with_atoms(argon.box_size, argon.pos.clone(), argon.v.clone(), argon.bath_temperature, false)
    }

    /// Hexatic order parameter: |mean over atoms of the mean of exp(6 i theta) over neighbours|.
    fn hexatic_order(argon: &Argon) -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for i in 0..argon.pos.len() {
            let (mut sum_re, mut sum_im, mut count) = (0.0, 0.0, 0);
            for j in 0..argon.pos.len() {
                if i == j {
                    continue;
                }
                let d = argon.minimum_image(argon.pos[j] - argon.pos[i]);
                if d.length() < NEIGHBOUR_DISTANCE {
                    let angle = 6.0 * d.y.atan2(d.x);
                    sum_re += angle.cos();
                    sum_im += angle.sin();
                    count += 1;
                }
            }
            if count > 0 {
                re += sum_re / count as f64;
                im += sum_im / count as f64;
            }
        }
        re.hypot(im) / argon.pos.len() as f64
    }

    #[test]
    fn unit_conversions() {
        let (epsilon, sigma, m) = (EPSILON_KELVIN * 1.380649e-23, SIGMA_ANGSTROM * 1e-10, MASS_U * 1.66053906660e-27);
        let tau_ps = sigma * (m / epsilon).sqrt() * 1e12;
        assert!((tau_ps / 2.156349 - 1.0).abs() < 1e-6, "tau {tau_ps} ps");
        assert!((tau_ps / TAU_PS - 1.0).abs() < 1e-6);
        let pressure = epsilon / (sigma * sigma) * 1e3;
        assert!((pressure / 14.26612 - 1.0).abs() < 1e-6, "pressure unit {pressure} mN/m");
        assert!((pressure / PRESSURE_MN_PER_M - 1.0).abs() < 1e-6);
    }

    #[test]
    fn force_is_minus_gradient_of_potential() {
        let u = |r: f64| pair(r * r).0;
        for r in [0.95, 1.0, R_MIN, 1.5, 2.5, 3.4] {
            let force = pair(r * r).1 / r;
            let h = 1e-5;
            let numeric = -(u(r + h) - u(r - h)) / (2.0 * h);
            assert!((force - numeric).abs() < 1e-6 * force.abs().max(1.0), "r = {r}: force {force}, gradient {numeric}");
        }
        assert!(pair(R_MIN * R_MIN).1.abs() < 1e-9);
        assert!((u(R_MIN) - (-1.0 - U_CUTOFF)).abs() < 1e-9);
        assert!(u(CUTOFF - 1e-9).abs() < 1e-8);
        assert!((U_CUTOFF - -0.0021747804).abs() < 1e-9, "u_c {U_CUTOFF}");
    }

    #[test]
    fn cell_list_matches_all_pairs() {
        let box_size = dvec2(30.0, 20.0);
        let rng = RandGenerator::new();
        rng.srand(5);
        let mut pos: Vec<DVec2> = Vec::new();
        while pos.len() < 300 {
            let p = dvec2(rng.gen_range(0.0_f64, 30.0), rng.gen_range(0.0_f64, 20.0));
            let argon = Argon::with_atoms(box_size, vec![], vec![], 1.0, false);
            if pos.iter().all(|q| argon.minimum_image(p - *q).length() > 0.9) {
                pos.push(p);
            }
        }
        let argon = Argon::with_atoms(box_size, pos.clone(), vec![DVec2::ZERO; 300], 1.0, false);

        // Direct double loop.
        let mut force = vec![DVec2::ZERO; 300];
        let mut coordination = vec![0u8; 300];
        let (mut energy, mut virial) = (0.0, 0.0);
        for i in 0..300 {
            for j in (i + 1)..300 {
                let d = argon.minimum_image(pos[i] - pos[j]);
                let r2 = d.length_squared();
                if r2 < NEIGHBOUR_DISTANCE * NEIGHBOUR_DISTANCE {
                    coordination[i] += 1;
                    coordination[j] += 1;
                }
                let (u, rf) = pair(r2);
                if rf != 0.0 {
                    force[i] += d * (rf / r2);
                    force[j] -= d * (rf / r2);
                    energy += u;
                    virial += rf;
                }
            }
        }
        assert!((argon.potential_energy - energy).abs() < 1e-9 * energy.abs().max(1.0), "energy {} vs {energy}", argon.potential_energy);
        assert!((argon.virial - virial).abs() < 1e-9 * virial.abs().max(1.0), "virial");
        assert_eq!(argon.coordination, coordination);
        for i in 0..300 {
            assert!((argon.force[i] - force[i]).length() < 1e-9 * force[i].length().max(1.0), "force on atom {i}");
        }
    }

    #[test]
    fn two_atoms_oscillate_at_the_harmonic_frequency() {
        let pos = vec![dvec2(6.0, 6.0), dvec2(6.0 + R_MIN + 0.001, 6.0)];
        let mut argon = Argon::with_atoms(dvec2(12.0, 12.0), pos, vec![DVec2::ZERO; 2], 1.0, false);
        let separation = |argon: &Argon| argon.minimum_image(argon.pos[1] - argon.pos[0]).x - R_MIN;
        let (mut crossings, mut first, mut last, mut t) = (0, 0.0, 0.0, 0.0);
        let mut previous = separation(&argon);
        while crossings < 20 {
            argon.step(DT);
            t += DT;
            let now = separation(&argon);
            if previous < 0.0 && now >= 0.0 {
                // Linear interpolation of the crossing inside the step.
                let crossing = t - DT * now / (now - previous);
                if crossings == 0 {
                    first = crossing;
                }
                last = crossing;
                crossings += 1;
            }
            previous = now;
        }
        // Spring constant u''(r_min) = 72 2^(-1/3), reduced mass 1/2.
        let expected = TAU / (2.0 * 72.0 * 2f64.powf(-1.0 / 3.0)).sqrt();
        let measured = (last - first) / 19.0;
        assert!((measured / expected - 1.0).abs() < 5e-4, "period {measured}, expected {expected}");
    }

    #[test]
    fn crystal_matches_lattice_sum() {
        let (columns, rows) = (12, 14);
        let pos = triangular_lattice(columns, rows, LATTICE_SPACING, DVec2::ZERO);
        let box_size = dvec2(columns as f64 * LATTICE_SPACING, rows as f64 * LATTICE_SPACING * 3f64.sqrt() / 2.0);
        let n = pos.len();
        let argon = Argon::with_atoms(box_size, pos, vec![DVec2::ZERO; n], 0.1, false);
        let per_atom = argon.energy() / n as f64;
        assert!((per_atom - -3.32529333).abs() < 1e-7, "energy per atom {per_atom}");
        assert!(argon.pressure().abs() < 1e-5, "pressure {}", argon.pressure());
        assert!(argon.coordination.iter().all(|&c| c == 6));
    }

    #[test]
    fn conserves_energy_and_momentum() {
        let mut argon = isolated_copy(&fluid());
        let (e0, p0, n) = (argon.energy(), argon.momentum(), 400.0);
        let (mut worst, mut first_half, mut second_half): (f64, f64, f64) = (0.0, 0.0, 0.0);
        for step in 0..20_000 {
            argon.step(DT);
            let e = (argon.energy() - e0) / n;
            worst = worst.max(e.abs());
            if step < 10_000 { first_half += e } else { second_half += e }
        }
        assert!(worst < 1e-3, "energy error {worst}");
        assert!((first_half - second_half).abs() / 10_000.0 < 1e-4, "energy drifts: {} vs {}", first_half / 1e4, second_half / 1e4);
        assert!((argon.momentum() - p0).length() < 1e-9, "momentum changed by {}", (argon.momentum() - p0).length());
    }

    #[test]
    fn energy_error_is_second_order() {
        let start = fluid();
        let worst = |dt: f64| {
            let mut argon = isolated_copy(&start);
            let e0 = argon.energy();
            let mut worst: f64 = 0.0;
            for _ in 0..(2.0 / dt).round() as usize {
                argon.step(dt);
                worst = worst.max((argon.energy() - e0).abs());
            }
            worst
        };
        let ratio = worst(0.005) / worst(0.0025);
        assert!(ratio > 3.5 && ratio < 4.5, "error ratio {ratio}");
    }

    #[test]
    fn collisions_produce_maxwell_boltzmann() {
        let isotropic = |rng: &RandGenerator| {
            let angle = TAU * rng.gen_range(0.0_f64, 1.0);
            dvec2(angle.cos(), angle.sin()) * 1.4f64.sqrt()
        };
        let mut argon = lattice_fluid(isotropic, false, 1.0, 2);
        let ratio = |argon: &Argon| {
            let mean_speed = argon.v.iter().map(|v| v.length()).sum::<f64>() / 400.0;
            let mean_square = argon.v.iter().map(|v| v.length_squared()).sum::<f64>() / 400.0;
            mean_speed * mean_speed / mean_square
        };
        assert!(ratio(&argon) > 0.99, "starts with ratio {}", ratio(&argon));
        run(&mut argon, 2000);

        let (mut ratio_sum, mut temperature_sum, mut samples) = (0.0, 0.0, 0.0);
        let mut kinetic_energies: Vec<Vec<f64>> = Vec::new();
        for step in 0..4000 {
            argon.step(DT);
            if step % 10 == 0 {
                ratio_sum += ratio(&argon);
                temperature_sum += argon.temperature();
                samples += 1.0;
                kinetic_energies.push(argon.v.iter().map(|v| 0.5 * v.length_squared()).collect());
            }
        }
        // 2D Maxwell-Boltzmann: <v>^2 / <v^2> = pi/4, and the fraction of atoms with energy below
        // x T is 1 - exp(-x).
        assert!((ratio_sum / samples - PI / 4.0).abs() < 0.006, "<v>^2/<v^2> = {}", ratio_sum / samples);
        let temperature = temperature_sum / samples;
        for x in [0.5, 1.0, 2.0] {
            let below = kinetic_energies.iter().flatten().filter(|&&e| e < x * temperature).count() as f64;
            let fraction = below / (400.0 * samples);
            assert!((fraction - (1.0 - (-x).exp())).abs() < 0.01, "fraction below {x} T: {fraction}");
        }
    }

    #[test]
    fn thermostat_holds_the_bath_temperature() {
        let mut argon = fluid();
        let mean_temperature = |argon: &mut Argon| {
            let mut sum = 0.0;
            for _ in 0..20_000 {
                argon.step(DT);
                sum += argon.temperature();
            }
            sum / 20_000.0
        };
        let first = mean_temperature(&mut argon);
        assert!((first / 0.7 - 1.0).abs() < 0.03, "mean T {first} at bath 0.7");
        argon.bath_temperature = 1.0;
        run(&mut argon, 4000);
        let second = mean_temperature(&mut argon);
        assert!((second / 1.0 - 1.0).abs() < 0.03, "mean T {second} at bath 1.0");
    }

    /// Second virial coefficient B2 = -pi integral of (exp(-u/T) - 1) r dr, by Simpson's rule.
    fn second_virial(t: f64) -> f64 {
        let n = 20_000;
        let h = CUTOFF / n as f64;
        let f = |r: f64| if r < 0.5 { -r } else { ((-pair(r * r).0 / t).exp() - 1.0) * r };
        let mut sum = f(0.0) + f(CUTOFF);
        for i in 1..n {
            sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        -PI * sum * h / 3.0
    }

    #[test]
    fn dilute_gas_follows_virial_expansion() {
        assert!((second_virial(1.0) - -1.00845).abs() < 1e-4, "B2(1) = {}", second_virial(1.0));
        assert!((second_virial(3.0) - 0.73450).abs() < 1e-4, "B2(3) = {}", second_virial(3.0));

        let side = (400.0f64 / 0.03).sqrt();
        let rho = 0.03;
        for (t, b3, expected, tolerance) in [(1.0, 2.4185, 0.97192, 0.004), (3.0, 1.1797, 1.02310, 0.003)] {
            // Random atoms, none closer than 1 sigma.
            let rng = RandGenerator::new();
            rng.srand(8);
            let mut pos: Vec<DVec2> = Vec::new();
            while pos.len() < 400 {
                let p = dvec2(rng.gen_range(0.0_f64, side), rng.gen_range(0.0_f64, side));
                let far = pos.iter().all(|q| {
                    let mut d = p - *q;
                    d -= (d / side).round() * side;
                    d.length() > 1.0
                });
                if far {
                    pos.push(p);
                }
            }
            let mut argon = Argon::with_atoms(dvec2(side, side), pos, vec![DVec2::ZERO; 400], t, true);
            argon.bath_temperature = t;
            run(&mut argon, 5000);
            let mut virial = 0.0;
            for _ in 0..100_000 {
                argon.step(DT);
                virial += argon.virial;
            }
            // Z = P A / (N T) = 1 + <W> / (2 N T) = 1 + B2 rho + B3 rho^2.
            let z = 1.0 + virial / 100_000.0 / (2.0 * 400.0 * t);
            let series = 1.0 + second_virial(t) * rho + b3 * rho * rho;
            assert!((series - expected).abs() < 1e-4, "series {series}");
            assert!((z - expected).abs() < tolerance, "T = {t}: Z = {z}, expected {expected}");
        }
    }

    #[test]
    fn crystal_survives_cold_and_melts_hot() {
        let order_after = |bath: f64, steps: usize| {
            let mut argon = Argon::new();
            argon.bath_temperature = bath;
            run(&mut argon, steps);
            hexatic_order(&argon)
        };
        let cold = order_after(0.30, 10_000);
        let hot = order_after(0.60, 10_000);
        assert!(cold > 0.7, "order at 36 K: {cold}");
        assert!(hot < 0.2, "order at 72 K: {hot}");
    }

    #[test]
    fn island_melts_between_43_and_53_kelvin() {
        let order_after = |bath: f64, steps: usize| {
            let mut argon = Argon::new();
            argon.bath_temperature = bath;
            run(&mut argon, steps);
            hexatic_order(&argon)
        };
        let below = order_after(0.36, 20_000);
        let above = order_after(0.44, 30_000);
        assert!(below > 0.5, "order at 0.36: {below}");
        assert!(above < 0.3, "order at 0.44: {above}");
    }
}
