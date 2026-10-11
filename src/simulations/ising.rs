use std::cell::OnceCell;
use std::collections::VecDeque;

use macroquad::prelude::*;
use macroquad::rand::RandGenerator;

use crate::simulation::Simulation;

// Units: reduced. The coupling J and Boltzmann's constant are 1, so a temperature `T` means
// k_B T / J, an energy per spin `e` means E / (N J), and a field `h` means (moment x field) / J.
// Time is counted in sweeps (N attempted flips) and is not a physical time.

const LATTICE_SIZE: usize = 128;
/// Onsager's critical temperature: sinh(2/T_c) = 1.
const T_CRITICAL: f64 = 2.269_185_314_213_022; // 2 / ln(1 + sqrt 2)
/// Below the critical temperature, so unpausing shows domains forming.
const T_START: f64 = 2.0;
const T_MIN: f64 = 0.1;
const T_MAX: f64 = 4.0;
/// Change of temperature per second while a key is held.
const T_RATE: f64 = 0.5;
const H_MAX: f64 = 1.0;
const H_RATE: f64 = 0.2;
/// Sweeps per second of real time.
const START_SPEED: f64 = 60.0;
const MIN_SPEED: f64 = 7.5;
const MAX_SPEED: f64 = 240.0;
/// Chosen so the first quench from random spins orders (the lattice can also freeze into stripes).
const SEED: u64 = 4;

const SPIN_UP: [u8; 4] = [240, 200, 70, 255];
const SPIN_DOWN: [u8; 4] = [25, 45, 90, 255];
/// Where the plot of |m| against T is drawn, in world units.
const PLOT: Rect = Rect { x: -0.48, y: 0.08, w: 0.42, h: 0.32 };
/// One point in the plot's history every this many sweeps, and how many are kept.
const HISTORY_EVERY: u64 = 5;
const HISTORY_LEN: usize = 600;

/// Magnetisation per spin of the infinite lattice at zero field (Onsager 1949, Yang 1952).
fn exact_magnetisation(t: f64) -> f64 {
    if t >= T_CRITICAL {
        return 0.0;
    }
    (1.0 - (2.0 / t).sinh().powi(-4)).powf(0.125)
}

/// Energy per spin of the infinite lattice at zero field (Onsager 1944):
///   e = -coth(2K) [1 + (2/pi) q K1(kappa)],  K = 1/T,  q = 2 tanh(2K)^2 - 1,
/// with K1 the complete elliptic integral of the first kind. Since sqrt(1 - kappa^2) = |q|,
/// K1 = pi / (2 AGM(1, |q|)), which gives the form below.
fn exact_energy(t: f64) -> f64 {
    let two_k = 2.0 / t;
    let coth = 1.0 / two_k.tanh();
    let q = 2.0 * two_k.tanh().powi(2) - 1.0;
    if q.abs() < 1e-14 {
        return -coth; // exactly at T_c the integral diverges but q K1 goes to 0
    }
    // Arithmetic-geometric mean.
    let (mut a, mut b) = (1.0, q.abs());
    while (a - b).abs() >= 1e-15 * a {
        (a, b) = ((a + b) / 2.0, (a * b).sqrt());
    }
    -coth * (1.0 + q / a)
}

enum Start {
    Random,
    /// Only the tests start ordered.
    #[cfg_attr(not(test), allow(dead_code))]
    AllUp,
}

/// The two-dimensional Ising ferromagnet: spins that are up or down on a square lattice with
/// periodic boundaries, each wanting to line up with its four neighbours, flipped at random by
/// heat (Metropolis single-spin updates, one step is one sweep of N attempts). Arrow keys change
/// the temperature and the magnetic field. Below the critical temperature the spins order into
/// domains, above it they are noise, and the HUD shows Onsager's exact values for comparison.
/// Not modelled: real magnets (3D, quantum, spins in any direction), real dynamics (only
/// equilibrium averages are exact), an infinite lattice (the transition is rounded at L = 128),
/// and anything beyond the instantaneous values (no error bars or susceptibility).
pub struct Ising {
    size: usize,
    /// +1 or -1, index = y * size + x.
    spins: Vec<i8>,
    temperature: f64,
    field: f64,
    /// Flip probability min(1, exp(-dE/T)), indexed `[(s + 1) / 2][(n + 4) / 2]` for a spin `s`
    /// whose neighbours sum to `n`. Rebuilt when the temperature or field changes, so there is
    /// no `exp` in the inner loop.
    accept: [[f64; 5]; 2],
    /// Sum of all spins and sum of s_i s_j over all bonds, updated on every flip.
    spin_sum: i64,
    bond_sum: i64,
    rng: RandGenerator,
    sweeps: u64,
    /// Sweeps per second.
    speed: f64,
    /// Recent (temperature, |m|) for the plot.
    history: VecDeque<(f32, f32)>,
    /// Made on the first draw: it needs the window, which tests don't have.
    texture: OnceCell<Texture2D>,
}

impl Ising {
    pub fn new() -> Self {
        Self::with(LATTICE_SIZE, T_START, 0.0, Start::Random, SEED)
    }

    fn with(size: usize, temperature: f64, field: f64, start: Start, seed: u64) -> Self {
        let rng = RandGenerator::new();
        rng.srand(seed);
        let spins = match start {
            Start::AllUp => vec![1; size * size],
            Start::Random => (0..size * size).map(|_| if rng.gen_range(0.0_f64, 1.0) < 0.5 { 1 } else { -1 }).collect(),
        };
        let mut ising = Self {
            size,
            spins,
            temperature,
            field,
            accept: [[0.0; 5]; 2],
            spin_sum: 0,
            bond_sum: 0,
            rng,
            sweeps: 0,
            speed: START_SPEED,
            history: VecDeque::new(),
            texture: OnceCell::new(),
        };
        (ising.spin_sum, ising.bond_sum) = ising.recount();
        ising.set_conditions(temperature, field);
        ising
    }

    /// Sets the temperature and field (clamped to the allowed ranges) and rebuilds the table.
    fn set_conditions(&mut self, temperature: f64, field: f64) {
        self.temperature = temperature.clamp(T_MIN, T_MAX);
        self.field = field.clamp(-H_MAX, H_MAX);
        for (index, row) in self.accept.iter_mut().enumerate() {
            let s = 2.0 * index as f64 - 1.0;
            for (j, probability) in row.iter_mut().enumerate() {
                let n = 2.0 * j as f64 - 4.0;
                // Flipping changes the energy by dE = 2 s (n + h).
                *probability = (-2.0 * s * (n + self.field) / self.temperature).exp().min(1.0);
            }
        }
    }

    /// Spin sum and bond sum counted from scratch (each site's right and upper bond).
    fn recount(&self) -> (i64, i64) {
        let n = self.size;
        let (mut spin_sum, mut bond_sum) = (0, 0);
        for y in 0..n {
            for x in 0..n {
                let s = self.spins[y * n + x] as i64;
                let right = self.spins[y * n + (x + 1) % n] as i64;
                let up = self.spins[((y + 1) % n) * n + x] as i64;
                spin_sum += s;
                bond_sum += s * (right + up);
            }
        }
        (spin_sum, bond_sum)
    }

    fn spin_count(&self) -> f64 {
        (self.size * self.size) as f64
    }

    /// Magnetisation per spin, between -1 and 1.
    fn magnetisation(&self) -> f64 {
        self.spin_sum as f64 / self.spin_count()
    }

    /// Energy per spin.
    fn energy(&self) -> f64 {
        (-self.bond_sum as f64 - self.field * self.spin_sum as f64) / self.spin_count()
    }

    /// One sweep: N attempts, each at a random site, flipping with the Metropolis probability.
    /// Always draws two numbers per attempt (site, then acceptance) so runs are reproducible.
    fn sweep(&mut self) {
        let n = self.size;
        for _ in 0..n * n {
            let i = self.rng.gen_range(0, n * n);
            let r = self.rng.gen_range(0.0_f64, 1.0);
            let (x, y) = (i % n, i / n);
            let neighbours = self.spins[y * n + (x + 1) % n] as i64
                + self.spins[y * n + (x + n - 1) % n] as i64
                + self.spins[((y + 1) % n) * n + x] as i64
                + self.spins[((y + n - 1) % n) * n + x] as i64;
            let s = self.spins[i] as i64;
            if r < self.accept[((s + 1) / 2) as usize][((neighbours + 4) / 2) as usize] {
                self.spins[i] = -s as i8;
                self.spin_sum -= 2 * s;
                self.bond_sum -= 2 * s * neighbours;
            }
        }
    }

    /// Every spin random again: a fresh quench from infinite temperature.
    fn randomise(&mut self) {
        for spin in &mut self.spins {
            *spin = if self.rng.gen_range(0.0_f64, 1.0) < 0.5 { 1 } else { -1 };
        }
        (self.spin_sum, self.bond_sum) = self.recount();
        self.history.clear();
    }
}

impl Simulation for Ising {
    fn step(&mut self, _dt: f64) {
        self.sweep();
        self.sweeps += 1;
        if self.sweeps % HISTORY_EVERY == 0 {
            if self.history.len() == HISTORY_LEN {
                self.history.pop_front();
            }
            self.history.push_back((self.temperature as f32, self.magnetisation().abs() as f32));
        }
    }

    /// One sweep per step.
    fn dt(&self) -> f64 {
        1.0
    }

    /// Sweeps per second.
    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn is_over(&self) -> bool {
        false
    }

    fn world_bounds(&self) -> Rect {
        // The lattice is the unit square; the extra column on its left is for the HUD and plot.
        Rect::new(-0.52, -0.0135, 1.54, 1.027)
    }

    fn draw(&self) {
        // Lattice, as a texture with one pixel per spin.
        let n = self.size;
        let mut pixels = Vec::with_capacity(4 * n * n);
        for &spin in &self.spins {
            pixels.extend_from_slice(if spin > 0 { &SPIN_UP } else { &SPIN_DOWN });
        }
        let texture = self.texture.get_or_init(|| {
            let texture = Texture2D::from_rgba8(n as u16, n as u16, &pixels);
            texture.set_filter(FilterMode::Nearest);
            texture
        });
        texture.update_from_bytes(n as u32, n as u32, &pixels);
        draw_texture_ex(texture, 0.0, 0.0, WHITE, DrawTextureParams { dest_size: Some(vec2(1.0, 1.0)), ..Default::default() });

        // Plot of |m| against T: frame, T_c, Onsager's curve, and where the simulation has been.
        let to_plot = |t: f64, m: f64| vec2(PLOT.x + PLOT.w * (t / T_MAX) as f32, PLOT.y + PLOT.h * m as f32);
        let width = 0.004;
        draw_rectangle_lines(PLOT.x, PLOT.y, PLOT.w, PLOT.h, width, DARKGRAY);
        let (a, b) = (to_plot(T_CRITICAL, 0.0), to_plot(T_CRITICAL, 1.0));
        draw_line(a.x, a.y, b.x, b.y, width, DARKGRAY);
        // Sampled in m, not T, so the steep part near T_c gets enough points.
        let curve: Vec<Vec2> = (0..=100)
            .map(|k| {
                let m = k as f64 / 100.0;
                let t = if k == 100 { 0.0 } else { 2.0 / (1.0 - m.powi(8)).powf(-0.25).asinh() };
                to_plot(t, m)
            })
            .collect();
        for pair in curve.windows(2) {
            draw_line(pair[0].x, pair[0].y, pair[1].x, pair[1].y, width, WHITE);
        }
        let up = Color::from_rgba(SPIN_UP[0], SPIN_UP[1], SPIN_UP[2], 255);
        let count = self.history.len();
        for (i, &(t, m)) in self.history.iter().enumerate() {
            let p = to_plot(t as f64, m as f64);
            let fade = Color::new(up.r, up.g, up.b, (i + 1) as f32 / count as f32);
            draw_rectangle(p.x - 0.003, p.y - 0.003, 0.006, 0.006, fade);
        }
        let now = to_plot(self.temperature, self.magnetisation().abs());
        draw_circle(now.x, now.y, 0.008, up);
    }

    fn handle_input(&mut self) {
        let dt = get_frame_time() as f64;
        let (mut t, mut h) = (self.temperature, self.field);
        if is_key_down(KeyCode::Up) {
            t += T_RATE * dt;
        }
        if is_key_down(KeyCode::Down) {
            t -= T_RATE * dt;
        }
        if is_key_down(KeyCode::Right) {
            h += H_RATE * dt;
        }
        if is_key_down(KeyCode::Left) {
            h -= H_RATE * dt;
        }
        if is_key_pressed(KeyCode::C) {
            t = T_CRITICAL;
        }
        if is_key_pressed(KeyCode::Z) {
            h = 0.0;
        }
        if t != self.temperature || h != self.field {
            self.set_conditions(t, h);
        }
        if is_key_pressed(KeyCode::F) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::S) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
        if is_key_pressed(KeyCode::R) {
            self.randomise();
        }
    }

    fn hud(&self) -> Vec<String> {
        // The column is 304 px wide at 12.25 px per character, so lines stay under 23 characters.
        let mut lines = vec![
            format!("kT/J = {:.3}", self.temperature),
            format!("T/Tc = {:.3}", self.temperature / T_CRITICAL),
            format!("h/J = {:+.2}", self.field),
        ];
        if self.field == 0.0 {
            lines.push(format!("m = {:+.3} ({:.3})", self.magnetisation(), exact_magnetisation(self.temperature)));
            lines.push(format!("e/J = {:+.3} ({:+.3})", self.energy(), exact_energy(self.temperature)));
            lines.push("(Onsager, L = inf)".to_string());
        } else {
            lines.push(format!("m = {:+.3}", self.magnetisation()));
            lines.push(format!("e/J = {:+.3}", self.energy()));
        }
        lines.push(format!("sweep {}, {}/s", self.sweeps, self.speed));
        lines.push("UP/DOWN: T, C: Tc".to_string());
        lines.push("LEFT/RIGHT: h, Z: 0".to_string());
        lines.push("F/S: speed, R: random".to_string());
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Means over `sweeps` sweeps (after `equilibration`) of e, m, |m|, m^2 and m^4, sampled once per sweep.
    struct Averages {
        e: f64,
        m: f64,
        abs_m: f64,
        m2: f64,
        m4: f64,
    }

    fn averages(ising: &mut Ising, equilibration: usize, sweeps: usize) -> Averages {
        for _ in 0..equilibration {
            ising.sweep();
        }
        let mut sum = Averages { e: 0.0, m: 0.0, abs_m: 0.0, m2: 0.0, m4: 0.0 };
        for _ in 0..sweeps {
            ising.sweep();
            let m = ising.magnetisation();
            sum.e += ising.energy();
            sum.m += m;
            sum.abs_m += m.abs();
            sum.m2 += m * m;
            sum.m4 += m.powi(4);
        }
        let k = sweeps as f64;
        Averages { e: sum.e / k, m: sum.m / k, abs_m: sum.abs_m / k, m2: sum.m2 / k, m4: sum.m4 / k }
    }

    #[test]
    fn exact_formulas_match_reference_table() {
        // (T, e, m) from Onsager's formulas, checked against numerical integration.
        let table = [
            (1.0, -1.997160204112, 0.999275751957),
            (1.5, -1.951116573078, 0.986499602621),
            (2.0, -1.745564575313, 0.911319377877),
            (2.2, -1.546489144098, 0.784755131384),
            (T_CRITICAL, -1.414213562373, 0.0),
            (2.4, -1.203974467915, 0.0),
            (2.5, -1.106079203746, 0.0),
            (3.0, -0.817309592502, 0.0),
            (4.0, -0.557272240628, 0.0),
        ];
        for (t, e, m) in table {
            assert!((exact_energy(t) - e).abs() < 1e-11, "e({t}) = {}", exact_energy(t));
            assert!((exact_magnetisation(t) - m).abs() < 1e-11, "m({t}) = {}", exact_magnetisation(t));
        }
        assert_eq!(exact_magnetisation(T_CRITICAL), 0.0);
    }

    #[test]
    fn known_states_have_known_energy() {
        let aligned = Ising::with(8, 2.0, 0.3, Start::AllUp, 1);
        assert!((aligned.energy() - -2.3).abs() < 1e-12 && (aligned.magnetisation() - 1.0).abs() < 1e-12);

        // Checkerboard: every bond joins opposite spins.
        let mut board = Ising::with(8, 2.0, 0.0, Start::AllUp, 1);
        for (i, spin) in board.spins.iter_mut().enumerate() {
            if (i % 8 + i / 8) % 2 == 1 {
                *spin = -1;
            }
        }
        (board.spin_sum, board.bond_sum) = board.recount();
        assert!((board.energy() - 2.0).abs() < 1e-12 && board.magnetisation().abs() < 1e-12);
    }

    #[test]
    fn acceptance_table_is_metropolis() {
        let ising = Ising::with(4, 2.0, 0.0, Start::AllUp, 1);
        let up = &ising.accept[1]; // an up spin, by neighbour sum -4, -2, 0, 2, 4
        assert!((up[4] - (-4.0f64).exp()).abs() < 1e-15 && (up[4] - 0.018315638888734).abs() < 1e-15);
        assert!((up[3] - (-2.0f64).exp()).abs() < 1e-15 && (up[3] - 0.135335283236613).abs() < 1e-15);
        assert_eq!(&up[0..3], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tracked_sums_match_recount() {
        let mut ising = Ising::with(16, 2.5, 0.3, Start::Random, 6);
        for _ in 0..200 {
            ising.sweep();
        }
        assert_eq!((ising.spin_sum, ising.bond_sum), ising.recount());
    }

    #[test]
    fn samples_boltzmann_on_4x4() {
        // All 65,536 states of the 4x4 torus can be enumerated, so there is no finite-size error.
        let mut ising = Ising::with(4, T_CRITICAL, 0.0, Start::Random, 1);
        let a = averages(&mut ising, 1_000, 1_000_000);
        assert!((a.e - -1.565623787638).abs() < 0.007, "e {}", a.e);
        assert!((a.abs_m - 0.843860444813).abs() < 0.0035, "|m| {}", a.abs_m);
        assert!((a.m2 - 0.761358908569).abs() < 0.0045, "m^2 {}", a.m2);
    }

    #[test]
    fn samples_boltzmann_on_4x4_in_a_field() {
        let mut ising = Ising::with(4, 2.5, 0.2, Start::Random, 2);
        let a = averages(&mut ising, 1_000, 1_000_000);
        assert!((a.e - -1.617445339186).abs() < 0.011, "e {}", a.e);
        assert!((a.m - 0.650435840434).abs() < 0.017, "m {}", a.m);
    }

    #[test]
    fn ordered_phase_matches_onsager() {
        // Starts all up so it cannot get stuck in a stripe state.
        let mut ising = Ising::with(32, 2.0, 0.0, Start::AllUp, 3);
        let a = averages(&mut ising, 1_000, 20_000);
        assert!((a.e - exact_energy(2.0)).abs() < 0.006, "e {}", a.e);
        assert!((a.abs_m - exact_magnetisation(2.0)).abs() < 0.004, "|m| {}", a.abs_m);
    }

    #[test]
    fn disordered_phase_matches_onsager() {
        let mut ising = Ising::with(32, 3.0, 0.0, Start::Random, 4);
        let a = averages(&mut ising, 1_000, 20_000);
        assert!((a.e - exact_energy(3.0)).abs() < 0.005, "e {}", a.e);
        // The exact magnetisation is 0, but a finite lattice shows noise of order 1/L.
        assert!(a.abs_m < 0.10, "|m| {}", a.abs_m);
    }

    #[test]
    fn critical_point_matches_exact_finite_lattice() {
        // Kaufman's exact energy of the 16x16 torus at T_c, and the Binder cumulant there
        // (Kamieniarz and Bloete 1993).
        let mut ising = Ising::with(16, T_CRITICAL, 0.0, Start::Random, 5);
        let a = averages(&mut ising, 2_000, 200_000);
        assert!((a.e - -1.4530649).abs() < 0.013, "e {}", a.e);
        let binder = 1.0 - a.m4 / (3.0 * a.m2 * a.m2);
        assert!((binder - 0.61069).abs() < 0.012, "U4 {binder}");
    }
}
