use std::cell::OnceCell;
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::ops::{Add, Mul, Sub};

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: length in nm, time in fs, energy in eV. Mass is then in eV fs^2 / nm^2.
/// Reduced Planck constant, eV fs: h / (2 pi e), exact in the SI.
const HBAR: f64 = 0.658_211_956_9;
const ELECTRON_MC2: f64 = 510_998.950; // eV, CODATA 2018
const C: f64 = 299.792_458; // nm/fs, exact
/// Electron mass.
const M: f64 = ELECTRON_MC2 / (C * C);

const NX: usize = 512;
const NY: usize = 256;
/// Grid spacing, nm: a 64 x 32 nm periodic box.
const DX: f64 = 0.125;
/// Absorbing layer inside each edge: width (nm) and the strength of its imaginary potential (eV).
const ABSORBER_WIDTH: f64 = 4.0;
const ABSORBER_STRENGTH: f64 = 2.0;
/// The detector screen is the grid column nearest above this x (nm).
const SCREEN_X: f64 = 27.5;
/// Steps per frame at the start speed.
const STEPS_PER_FRAME: f64 = 2.0;
/// One shot stands for this many identically prepared electrons shown at once.
const ELECTRONS_PER_SHOT: f64 = 2000.0;
const MAX_DOTS: usize = 6000;
/// Single electrons: arrivals per second of real time, the start and the choices (+/-).
const START_RATE: f64 = 3.0;
const RATES: [f64; 7] = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0];
/// A shot is over when less than this much probability is left on the grid.
const EMPTY: f64 = 1e-3;
const BRIGHTNESS_GAIN: f64 = 1.5;

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    fn from_phase(angle: f64) -> Self {
        Self { re: angle.cos(), im: angle.sin() }
    }

    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn scale(self, k: f64) -> Self {
        Self { re: self.re * k, im: self.im * k }
    }

    fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex { re: self.re + o.re, im: self.im + o.im }
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex { re: self.re - o.re, im: self.im - o.im }
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

/// Iterative in-place radix-2 FFT (Cooley-Tukey) for one length. Forward sign is e^(-2 pi i jk/n);
/// the inverse divides by n.
struct Fft {
    twiddles: Vec<Complex>,
    bit_reversed: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT length must be a power of two");
        let bits = n.trailing_zeros();
        Self {
            twiddles: (0..n / 2).map(|k| Complex::from_phase(-TAU * k as f64 / n as f64)).collect(),
            bit_reversed: (0..n).map(|i| if n == 1 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) }).collect(),
        }
    }

    fn transform(&self, data: &mut [Complex], inverse: bool) {
        let n = data.len();
        for i in 0..n {
            let j = self.bit_reversed[i];
            if j > i {
                data.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let mut w = self.twiddles[k * n / len];
                    if inverse {
                        w = w.conj();
                    }
                    let odd = data[start + k + len / 2] * w;
                    let even = data[start + k];
                    data[start + k] = even + odd;
                    data[start + k + len / 2] = even - odd;
                }
            }
            len *= 2;
        }
        if inverse {
            for value in data.iter_mut() {
                *value = value.scale(1.0 / n as f64);
            }
        }
    }
}

/// A Gaussian wave packet moving in +x: centre, standard deviations of |psi|^2 (nm) and kinetic
/// energy (eV).
#[derive(Clone, Copy)]
struct Packet {
    x0: f64,
    y0: f64,
    sigma_x: f64,
    sigma_y: f64,
    energy: f64,
}

impl Packet {
    fn k0(&self) -> f64 {
        (2.0 * M * self.energy).sqrt() / HBAR
    }
}

/// Wave number of FFT index `j` on an axis of `n` points.
fn wave_number(j: usize, n: usize, dx: f64) -> f64 {
    let j = if j < n / 2 { j as f64 } else { j as f64 - n as f64 };
    TAU * j / (n as f64 * dx)
}

/// Coordinate of grid index `i` on an axis of `n` points, centred on 0.
fn coordinate(i: usize, n: usize, dx: f64) -> f64 {
    (i as f64 - (n / 2) as f64 + 0.5) * dx
}

/// Imaginary part -W of the potential: a layer inside each edge that swallows the wave,
/// W = W0 s^3 with s the depth into the layer. `ny == 1` is a 1D grid.
fn absorber(nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    let depth = |position: f64, half_size: f64| ((position.abs() - (half_size - ABSORBER_WIDTH)) / ABSORBER_WIDTH).max(0.0);
    let mut w = vec![0.0; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            let sx = depth(coordinate(ix, nx, dx), nx as f64 * dx / 2.0);
            let sy = if ny == 1 { 0.0 } else { depth(coordinate(iy, ny, dx), ny as f64 * dx / 2.0) };
            w[iy * nx + ix] = ABSORBER_STRENGTH * (sx.powi(3) + sy.powi(3));
        }
    }
    w
}

/// The solver of the time-dependent Schroedinger equation on a periodic grid (no macroquad
/// calls, so tests can build it at any size, including `ny = 1`).
///   i hbar dpsi/dt = -(hbar^2 / 2m) laplacian(psi) + (V - iW) psi
/// Split-operator Fourier method (Strang): half a step of the potential, a full step of the
/// kinetic energy in Fourier space (the exact free propagator, so no numerical dispersion), half a
/// step of the potential. Every factor has modulus 1 without the absorber, so the norm is
/// conserved to rounding and there is no stability limit; the error is O(dt^2), from the
/// splitting, and only where V is not zero.
struct Wave {
    nx: usize,
    ny: usize,
    dx: f64,
    dt: f64,
    t: f64,
    /// Index `iy * nx + ix`.
    psi: Vec<Complex>,
    /// Only the tests read it back (for the energy).
    #[cfg_attr(not(test), allow(dead_code))]
    potential: Vec<f64>,
    absorber: Vec<f64>,
    potential_factor: Vec<Complex>,
    kinetic_factor: Vec<Complex>,
    /// Probability removed by the absorber so far at each point.
    absorbed: Vec<f64>,
    fft_x: Fft,
    fft_y: Fft,
    column: Vec<Complex>,
}

impl Wave {
    fn new(nx: usize, ny: usize, dx: f64, dt: f64, potential: Vec<f64>, absorber: Vec<f64>) -> Self {
        // exp(-i (V - iW) dt / 2 hbar)
        let potential_factor = potential
            .iter()
            .zip(&absorber)
            .map(|(v, w)| Complex::from_phase(-v * dt / (2.0 * HBAR)).scale((-w * dt / (2.0 * HBAR)).exp()))
            .collect();
        let mut kinetic_factor = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                let ky = if ny == 1 { 0.0 } else { wave_number(iy, ny, dx) };
                let k2 = wave_number(ix, nx, dx).powi(2) + ky * ky;
                kinetic_factor.push(Complex::from_phase(-HBAR * k2 * dt / (2.0 * M)));
            }
        }
        Self {
            nx,
            ny,
            dx,
            dt,
            t: 0.0,
            psi: vec![Complex::ZERO; nx * ny],
            potential,
            absorber,
            potential_factor,
            kinetic_factor,
            absorbed: vec![0.0; nx * ny],
            fft_x: Fft::new(nx),
            fft_y: Fft::new(ny),
            column: vec![Complex::ZERO; ny],
        }
    }

    fn x(&self, ix: usize) -> f64 {
        coordinate(ix, self.nx, self.dx)
    }

    fn y(&self, iy: usize) -> f64 {
        coordinate(iy, self.ny, self.dx)
    }

    /// Area of a grid cell (length of one in 1D).
    fn cell(&self) -> f64 {
        if self.ny == 1 { self.dx } else { self.dx * self.dx }
    }

    /// Starts a packet: normalised so that the sum of |psi|^2 over the grid times the cell is 1.
    fn set_packet(&mut self, packet: &Packet) {
        let k0 = packet.k0();
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let dx = self.x(ix) - packet.x0;
                let mut exponent = -dx * dx / (4.0 * packet.sigma_x * packet.sigma_x);
                if self.ny > 1 {
                    let dy = self.y(iy) - packet.y0;
                    exponent -= dy * dy / (4.0 * packet.sigma_y * packet.sigma_y);
                }
                self.psi[iy * self.nx + ix] = Complex::from_phase(k0 * dx).scale(exponent.exp());
            }
        }
        let scale = 1.0 / self.norm().sqrt();
        for value in &mut self.psi {
            *value = value.scale(scale);
        }
        self.absorbed.fill(0.0);
        self.t = 0.0;
    }

    /// Multiplies by the potential factor, noting what the absorber takes.
    fn potential_half_step(&mut self) {
        let cell = self.cell();
        for i in 0..self.psi.len() {
            let before = self.psi[i].norm_sqr();
            self.psi[i] = self.psi[i] * self.potential_factor[i];
            if self.absorber[i] > 0.0 {
                self.absorbed[i] += (before - self.psi[i].norm_sqr()) * cell;
            }
        }
    }

    /// 2D FFT: every row, then every column through a scratch buffer.
    fn fft2(&mut self, inverse: bool) {
        for row in self.psi.chunks_mut(self.nx) {
            self.fft_x.transform(row, inverse);
        }
        if self.ny > 1 {
            for ix in 0..self.nx {
                for iy in 0..self.ny {
                    self.column[iy] = self.psi[iy * self.nx + ix];
                }
                self.fft_y.transform(&mut self.column, inverse);
                for iy in 0..self.ny {
                    self.psi[iy * self.nx + ix] = self.column[iy];
                }
            }
        }
    }

    fn step(&mut self) {
        self.potential_half_step();
        self.fft2(false);
        for (value, factor) in self.psi.iter_mut().zip(&self.kinetic_factor) {
            *value = *value * *factor;
        }
        self.fft2(true);
        self.potential_half_step();
        self.t += self.dt;
    }

    /// Probability left on the grid.
    fn norm(&self) -> f64 {
        self.psi.iter().map(|p| p.norm_sqr()).sum::<f64>() * self.cell()
    }

    /// Mean and standard deviation of x and of y.
    fn moments(&self) -> (f64, f64, f64, f64) {
        let cell = self.cell();
        let (mut sx, mut sxx, mut sy, mut syy, mut total) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let p = self.psi[iy * self.nx + ix].norm_sqr() * cell;
                let (x, y) = (self.x(ix), self.y(iy));
                total += p;
                sx += p * x;
                sxx += p * x * x;
                sy += p * y;
                syy += p * y * y;
            }
        }
        let (mx, my) = (sx / total, sy / total);
        (mx, (sxx / total - mx * mx).sqrt(), my, (syy / total - my * my).sqrt())
    }

    /// Probability in the columns for which `keep(ix)` is true, including what the absorber took there.
    fn probability_where(&self, keep: impl Fn(usize) -> bool) -> f64 {
        let cell = self.cell();
        let mut sum = 0.0;
        for iy in 0..self.ny {
            for ix in (0..self.nx).filter(|&ix| keep(ix)) {
                let i = iy * self.nx + ix;
                sum += self.psi[i].norm_sqr() * cell + self.absorbed[i];
            }
        }
        sum
    }

    fn total_absorbed(&self) -> f64 {
        self.absorbed.iter().sum()
    }

    /// Probability current in +x, (hbar/m) Im(psi* dpsi/dx), probability per nm per fs through a
    /// vertical line. The derivative is the fourth-order central difference.
    fn flux_x(&self, ix: usize, iy: usize) -> f64 {
        let at = |offset: isize| self.psi[iy * self.nx + (ix as isize + offset).rem_euclid(self.nx as isize) as usize];
        let derivative = (at(-2) - at(2) + (at(1) - at(-1)).scale(8.0)).scale(1.0 / (12.0 * self.dx));
        HBAR / M * (at(0).conj() * derivative).im
    }

    /// Expectation value of the energy: potential plus kinetic (from the Fourier transform).
    #[cfg(test)]
    fn energy(&self) -> f64 {
        let cell = self.cell();
        let potential: f64 = self.psi.iter().zip(&self.potential).map(|(p, v)| v * p.norm_sqr()).sum::<f64>() * cell;
        let mut copy = Wave::new(self.nx, self.ny, self.dx, self.dt, vec![0.0; self.nx * self.ny], vec![0.0; self.nx * self.ny]);
        copy.psi.clone_from(&self.psi);
        copy.fft2(false);
        let mut kinetic = 0.0;
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let ky = if self.ny == 1 { 0.0 } else { wave_number(iy, self.ny, self.dx) };
                let k2 = wave_number(ix, self.nx, self.dx).powi(2) + ky * ky;
                kinetic += HBAR * HBAR * k2 / (2.0 * M) * copy.psi[iy * self.nx + ix].norm_sqr();
            }
        }
        potential + kinetic * cell / (self.nx * self.ny) as f64
    }
}

/// A rectangle of constant potential, eV (nm).
#[derive(Clone, Copy)]
struct Block {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    height: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Free,
    Barrier,
    DoubleSlit,
}

struct Scene {
    kind: Kind,
    packet: Packet,
    dt: f64,
    blocks: Vec<Block>,
    /// The x that separates "reflected" from "transmitted".
    divider: f64,
}

const FAR: f64 = 1e9;

// The thin barrier: a wall 1 nm thick, 0.5 eV high, for an electron of 0.375 eV.
const BARRIER_HEIGHT: f64 = 0.5;
const BARRIER_WIDTH: f64 = 1.0;
// The double slit: a wall 0.5 nm thick, 12 eV high, with slits 1.25 nm wide centred on y = +-4.
const WALL_X: (f64, f64) = (-10.0, -9.5);
const SLIT_WIDTH: f64 = 1.25;
const SLIT_CENTRE: f64 = 4.0;
const WALL_HEIGHT: f64 = 12.0;

impl Scene {
    fn free() -> Self {
        Self {
            kind: Kind::Free,
            packet: Packet { x0: -20.0, y0: 0.0, sigma_x: 1.0, sigma_y: 1.0, energy: 1.5 },
            dt: 0.05,
            blocks: Vec::new(),
            divider: 0.0,
        }
    }

    fn barrier() -> Self {
        Self {
            kind: Kind::Barrier,
            packet: Packet { x0: -16.0, y0: 0.0, sigma_x: 4.0, sigma_y: 3.0, energy: 0.375 },
            dt: 0.1,
            blocks: vec![Block { x_min: 0.0, x_max: BARRIER_WIDTH, y_min: -FAR, y_max: FAR, height: BARRIER_HEIGHT }],
            divider: 0.5,
        }
    }

    fn double_slit() -> Self {
        // Three pieces of wall leave the two slits open.
        let (low, high) = (SLIT_CENTRE - SLIT_WIDTH / 2.0, SLIT_CENTRE + SLIT_WIDTH / 2.0);
        let piece = |y_min, y_max| Block { x_min: WALL_X.0, x_max: WALL_X.1, y_min, y_max, height: WALL_HEIGHT };
        Self {
            kind: Kind::DoubleSlit,
            packet: Packet { x0: -20.0, y0: 0.0, sigma_x: 2.0, sigma_y: 4.0, energy: 1.5 },
            dt: 0.05,
            blocks: vec![piece(-FAR, -high), piece(-low, low), piece(high, FAR)],
            divider: -9.75,
        }
    }

    /// Potential on the grid: the height of every block whose rectangle contains the point.
    fn potential(&self) -> Vec<f64> {
        let mut v = vec![0.0; NX * NY];
        for iy in 0..NY {
            for ix in 0..NX {
                let (x, y) = (coordinate(ix, NX, DX), coordinate(iy, NY, DX));
                for b in &self.blocks {
                    if x >= b.x_min && x <= b.x_max && y >= b.y_min && y <= b.y_max {
                        v[iy * NX + ix] = b.height;
                    }
                }
            }
        }
        v
    }
}

/// Transmission of a rectangular barrier of height `v0` and width `a` for energy `e`.
fn rectangular_barrier_transmission(e: f64, v0: f64, a: f64) -> f64 {
    if (e - v0).abs() < 1e-12 {
        return 1.0 / (1.0 + M * v0 * a * a / (2.0 * HBAR * HBAR));
    }
    if e < v0 {
        let kappa = (2.0 * M * (v0 - e)).sqrt() / HBAR;
        1.0 / (1.0 + v0 * v0 * (kappa * a).sinh().powi(2) / (4.0 * e * (v0 - e)))
    } else {
        let q = (2.0 * M * (e - v0)).sqrt() / HBAR;
        1.0 / (1.0 + v0 * v0 * (q * a).sin().powi(2) / (4.0 * e * (e - v0)))
    }
}

/// Transmission of a packet: the transmission at each wave number, weighted by the packet's
/// Gaussian distribution of wave numbers (sigma_k = 1 / 2 sigma_x), by the trapezoid rule over
/// k0 +- 8 sigma_k. A barrier that doesn't depend on y leaves the y motion free, so this is the
/// 2D result as well.
fn packet_transmission(packet: &Packet, transmission_at: impl Fn(f64) -> f64) -> f64 {
    let (k0, sigma_k) = (packet.k0(), 1.0 / (2.0 * packet.sigma_x));
    let n = 2000;
    let (low, high) = (k0 - 8.0 * sigma_k, k0 + 8.0 * sigma_k);
    let h = (high - low) / n as f64;
    let weighted = |k: f64| {
        let g = (-(k - k0).powi(2) / (2.0 * sigma_k * sigma_k)).exp() / (TAU * sigma_k * sigma_k).sqrt();
        g * transmission_at(HBAR * HBAR * k * k / (2.0 * M))
    };
    let mut sum = 0.5 * (weighted(low) + weighted(high));
    for i in 1..n {
        sum += weighted(low + i as f64 * h);
    }
    sum * h
}

/// y of the dark fringe `n` (n = 1/2, 3/2, ...) for two point sources a distance `d` apart and a
/// screen `l` beyond them, for wavelength `lambda`: path difference (n) lambda.
fn dark_fringe(n: f64, d: f64, l: f64, lambda: f64) -> f64 {
    let half_path = n * lambda / 2.0;
    half_path * (1.0 + l * l / ((d / 2.0).powi(2) - half_path.powi(2))).sqrt()
}

/// The dark fringes expected on the screen in the double-slit scene (n = 1/2 and 3/2).
/// Uses the slit separation reduced a little because the packet is brighter towards the axis, so
/// the inner half of each slit carries more amplitude: d_eff = d (1 - w^2 / 24 sigma_y^2).
fn expected_dark_fringes(scene: &Scene, screen_x: f64) -> [f64; 2] {
    let lambda = TAU / scene.packet.k0();
    let d = 2.0 * SLIT_CENTRE * (1.0 - SLIT_WIDTH.powi(2) / (24.0 * scene.packet.sigma_y.powi(2)));
    let l = screen_x - WALL_X.1;
    [dark_fringe(0.5, d, l, lambda), dark_fringe(1.5, d, l, lambda)]
}

/// Index of the weight that `pick` (between 0 and the sum of the positive weights) falls in, with
/// negative weights counted as 0.
fn pick_row(weights: &[f64], mut pick: f64) -> usize {
    for (i, weight) in weights.iter().enumerate() {
        pick -= weight.max(0.0);
        if pick <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// One electron, described by its wave function psi(x, y, t), fired as a Gaussian wave packet at
/// nothing (it drifts and spreads), a thin barrier higher than its energy (part tunnels through)
/// or a wall with two slits (the two waves interfere). Brightness is |psi|, colour the phase.
/// A detector screen on the right turns the probability arriving into dots that build up the
/// fringe pattern shot by shot. The time-dependent Schroedinger equation in 2D by the
/// split-operator Fourier method on a 512 x 256 grid with absorbing edges. Keys 1-3 choose the
/// scene.
/// E switches to single electrons, as in Tonomura's 1989 experiment: they arrive one at a time at
/// random moments, each makes one dot, and the fringes only emerge from many of them. Every
/// electron has the same wave function, so once the first wave has passed, the probability that
/// reached each row of the screen is where an electron lands; the ones that hit the wall or bounce
/// back are not seen, as in the real experiment.
/// Not modelled: real-scale double slits (the slits are 8 wavelengths apart, so the screen is in
/// the near field), the wall as a material, sharp edges finer than the 0.125 nm grid, collapse
/// (the dots are sampled from the probability flux; one shot stands for 2000 electrons shown at
/// once, or, with single electrons, the wave is computed once and each electron samples it), the
/// third dimension, spin and relativity (v/c is 0.0024), and open space (the box is periodic with
/// absorbing edges, 1e-7 to 1e-5 of the probability survives them).
pub struct WavePacket {
    scene: Scene,
    wave: Wave,
    speed: f64,
    /// Probability that has arrived at each row of the screen column.
    screen: Vec<f64>,
    screen_ix: usize,
    dots: VecDeque<Vec2>,
    dots_due: f64,
    detected: usize,
    shots: usize,
    /// One electron at a time instead of 2000 per shot.
    single: bool,
    /// Single electrons per second of real time, and the real time until the next one.
    rate: f64,
    next_electron_in: f64,
    /// Transmission of the barrier for this packet (closed form), for the HUD.
    exact_transmission: f64,
    /// Where the dark fringes should be on the screen (double slit).
    dark_fringes: [f64; 2],
    /// Largest |psi| at the start, for the brightness scale.
    peak: f64,
    texture: OnceCell<Texture2D>,
}

impl WavePacket {
    pub fn new() -> Self {
        Self::with_scene(Scene::free())
    }

    fn with_scene(scene: Scene) -> Self {
        Self::with_scene_and_mode(scene, false, START_RATE)
    }

    fn with_scene_and_mode(scene: Scene, single: bool, rate: f64) -> Self {
        let mut wave = Wave::new(NX, NY, DX, scene.dt, scene.potential(), absorber(NX, NY, DX));
        wave.set_packet(&scene.packet);
        let peak = wave.psi.iter().map(|p| p.norm_sqr().sqrt()).fold(0.0, f64::max);
        let screen_ix = (SCREEN_X / DX + (NX / 2) as f64 - 0.5).ceil() as usize;
        let exact_transmission = packet_transmission(&scene.packet, |e| rectangular_barrier_transmission(e, BARRIER_HEIGHT, BARRIER_WIDTH));
        let dark_fringes = expected_dark_fringes(&scene, wave.x(screen_ix));
        Self {
            speed: STEPS_PER_FRAME * 60.0 * scene.dt,
            scene,
            wave,
            screen: vec![0.0; NY],
            screen_ix,
            dots: VecDeque::new(),
            dots_due: 0.0,
            detected: 0,
            shots: 0,
            single,
            rate,
            next_electron_in: 0.0,
            exact_transmission,
            dark_fringes,
            peak,
            texture: OnceCell::new(),
        }
    }

    fn start_speed(&self) -> f64 {
        STEPS_PER_FRAME * 60.0 * self.scene.dt
    }

    fn add_dot(&mut self, row: usize) {
        let x = macroquad::rand::gen_range(SCREEN_X as f32, 32.0);
        let y = self.wave.y(row) as f32 + macroquad::rand::gen_range(-0.5, 0.5) * DX as f32;
        if self.dots.len() == MAX_DOTS {
            self.dots.pop_front();
        }
        self.dots.push_back(vec2(x, y));
        self.detected += 1;
    }

    /// Single electrons due in the `real_seconds` that just passed: Poisson arrivals at `rate`,
    /// each on a row drawn from the probability that reached the screen during the first shot.
    fn fire_single_electrons(&mut self, real_seconds: f64) {
        if self.shots == 0 {
            return; // the pattern isn't known until the first wave has passed
        }
        let total: f64 = self.screen.iter().map(|p| p.max(0.0)).sum();
        self.next_electron_in -= real_seconds;
        while self.next_electron_in <= 0.0 {
            let row = pick_row(&self.screen, macroquad::rand::gen_range(0.0, total));
            self.add_dot(row);
            // Exponential waiting time with mean 1 / rate.
            let u: f64 = macroquad::rand::gen_range(0.0, 1.0);
            self.next_electron_in += -(1.0 - u).ln() / self.rate;
        }
    }

    /// Probability left of the divider (reflected) and right of it (transmitted).
    fn split(&self) -> (f64, f64) {
        let divider = self.scene.divider;
        let transmitted = self.wave.probability_where(|ix| self.wave.x(ix) > divider);
        let reflected = self.wave.probability_where(|ix| self.wave.x(ix) <= divider);
        (reflected, transmitted)
    }
}

impl Simulation for WavePacket {
    fn step(&mut self, dt: f64) {
        debug_assert!((dt - self.wave.dt).abs() < 1e-12, "the tables are built for the scene's step");
        self.wave.step();

        // What reaches the detector column this step, row by row, and the dots it makes.
        let mut arrivals = [0.0; NY];
        for (iy, arrived) in arrivals.iter_mut().enumerate() {
            *arrived = self.wave.flux_x(self.screen_ix, iy) * self.wave.dt * DX;
            self.screen[iy] += *arrived;
        }
        if self.single {
            self.fire_single_electrons(dt / self.speed);
        } else {
            let total: f64 = arrivals.iter().map(|a| a.max(0.0)).sum();
            self.dots_due += ELECTRONS_PER_SHOT * total;
            while self.dots_due >= 1.0 {
                self.dots_due -= 1.0;
                let row = pick_row(&arrivals, macroquad::rand::gen_range(0.0, total));
                self.add_dot(row);
            }
        }

        // Fire again once nothing is left; the screen and the dots stay, so the pattern builds up.
        if self.wave.norm() < EMPTY {
            self.shots += 1;
            self.wave.set_packet(&self.scene.packet);
        }
    }

    fn is_over(&self) -> bool {
        false
    }

    fn dt(&self) -> f64 {
        self.scene.dt
    }

    /// fs per second of real time.
    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn world_bounds(&self) -> Rect {
        // The grid fills the lower part of the window; the top is for the five HUD lines.
        Rect::new(-35.0, -16.0, 70.0, 46.67)
    }

    fn handle_input(&mut self) {
        for (key, scene) in [(KeyCode::Key1, Scene::free as fn() -> Scene), (KeyCode::Key2, Scene::barrier), (KeyCode::Key3, Scene::double_slit)] {
            if is_key_pressed(key) {
                *self = Self::with_scene_and_mode(scene(), self.single, self.rate);
            }
        }
        if is_key_pressed(KeyCode::E) {
            // The two kinds of dots don't mix: start the screen's dots again.
            self.single = !self.single;
            self.dots.clear();
            self.detected = 0;
            self.dots_due = 0.0;
            self.next_electron_in = 0.0;
        }
        if self.single {
            let current = RATES.iter().position(|&r| r == self.rate).unwrap_or(1);
            if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
                self.rate = RATES[(current + 1).min(RATES.len() - 1)];
            }
            if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
                self.rate = RATES[current.saturating_sub(1)];
            }
        }
        let start = self.start_speed();
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(start * 4.0);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(start / 4.0);
        }
    }

    fn draw(&self) {
        // Colours: brightness is |psi|, hue is the phase (0 red, 120 degrees green, 240 blue),
        // without trigonometry per pixel.
        let mut pixels = Vec::with_capacity(4 * NX * NY);
        for p in &self.wave.psi {
            let magnitude = p.norm_sqr().sqrt();
            let b = (BRIGHTNESS_GAIN * magnitude / self.peak).min(1.0);
            let (c, s) = if magnitude > 0.0 { (p.re / magnitude, p.im / magnitude) } else { (1.0, 0.0) };
            let r = b * (1.0 + c) / 2.0;
            let g = b * (1.0 - c / 2.0 + 0.866 * s) / 2.0;
            let blue = b * (1.0 - c / 2.0 - 0.866 * s) / 2.0;
            pixels.extend_from_slice(&[(r * 255.0) as u8, (g * 255.0) as u8, (blue * 255.0) as u8, 255]);
        }
        let texture = self.texture.get_or_init(|| {
            let texture = Texture2D::from_rgba8(NX as u16, NY as u16, &pixels);
            texture.set_filter(FilterMode::Linear);
            texture
        });
        texture.update_from_bytes(NX as u32, NY as u32, &pixels);
        // Row 0 of the texture is at the bottom, like iy = 0.
        draw_texture_ex(texture, -32.0, -16.0, WHITE, DrawTextureParams { dest_size: Some(vec2(64.0, 32.0)), ..Default::default() });

        // Potential blocks, clipped to the box, and the edge of the absorbing layer.
        for b in &self.scene.blocks {
            let (x0, x1) = (b.x_min.max(-32.0) as f32, b.x_max.min(32.0) as f32);
            let (y0, y1) = (b.y_min.max(-16.0) as f32, b.y_max.min(16.0) as f32);
            draw_rectangle(x0, y0, x1 - x0, y1 - y0, Color::new(0.9, 0.9, 0.9, 0.35));
            draw_rectangle_lines(x0, y0, x1 - x0, y1 - y0, 0.1, WHITE);
        }
        let edge = ABSORBER_WIDTH as f32;
        draw_rectangle_lines(-32.0 + edge, -16.0 + edge, 64.0 - 2.0 * edge, 32.0 - 2.0 * edge, 0.08, DARKGRAY);

        // Detector screen: strip, line, dots, and the arrivals so far as a yellow curve.
        draw_rectangle(SCREEN_X as f32, -16.0, 32.0 - SCREEN_X as f32, 32.0, Color::new(0.08, 0.08, 0.1, 1.0));
        draw_line(SCREEN_X as f32, -16.0, SCREEN_X as f32, 16.0, 0.1, WHITE);
        for dot in &self.dots {
            draw_rectangle(dot.x - 0.075, dot.y - 0.075, 0.15, 0.15, WHITE);
        }
        // With single electrons, the newest one stands out.
        if self.single && let Some(dot) = self.dots.back() {
            draw_circle(dot.x, dot.y, 0.35, YELLOW);
        }
        let most = self.screen.iter().cloned().fold(0.0, f64::max);
        if most > 0.0 {
            for iy in 1..NY {
                let point = |iy: usize| vec2(SCREEN_X as f32 + 4.0 * (self.screen[iy].max(0.0) / most) as f32, self.wave.y(iy) as f32);
                let (p0, p1) = (point(iy - 1), point(iy));
                draw_line(p0.x, p0.y, p1.x, p1.y, 0.08, YELLOW);
            }
        }
        if self.scene.kind == Kind::DoubleSlit {
            for fringe in self.dark_fringes {
                for y in [fringe, -fringe] {
                    draw_line(SCREEN_X as f32 - 0.6, y as f32, SCREEN_X as f32, y as f32, 0.1, RED);
                }
            }
        }
    }

    fn hud(&self) -> Vec<String> {
        let packet = &self.scene.packet;
        let k0 = packet.k0();
        let v = HBAR * k0 / M;
        let third = match self.scene.kind {
            Kind::Free => {
                if self.wave.total_absorbed() > 1e-9 {
                    String::new()
                } else {
                    let (x, sigma_x, _, _) = self.wave.moments();
                    let spread = (1.0 + (HBAR * self.wave.t / (2.0 * M * packet.sigma_x.powi(2))).powi(2)).sqrt();
                    format!(
                        "x = {:.2} nm (exact {:.2})   width = {:.3} nm (exact {:.3})",
                        x,
                        packet.x0 + v * self.wave.t,
                        sigma_x,
                        packet.sigma_x * spread
                    )
                }
            }
            Kind::Barrier => {
                let (reflected, transmitted) = self.split();
                format!("reflected {:.4}  transmitted {:.4}  (exact for this packet {:.4})", reflected, transmitted, self.exact_transmission)
            }
            Kind::DoubleSlit => {
                let (_, through) = self.split();
                format!("through the slits {:.4}   dark fringes expected at {:.2}, {:.2} nm", through, self.dark_fringes[0], self.dark_fringes[1])
            }
        };
        let electrons = if !self.single {
            format!("{} electrons per shot (E: single electrons)", ELECTRONS_PER_SHOT)
        } else if self.shots == 0 {
            "single electrons (E): they start once the first wave has passed the screen".to_string()
        } else {
            format!("single electrons (E): {} per second (+/-)", self.rate)
        };
        vec![
            format!("[1] free  [2] barrier  [3] double slit   speed {:.0} fs/s (UP/DOWN)", self.speed),
            electrons,
            format!("t = {:.1} fs   E = {} eV   wavelength = {:.3} nm   v = {:.3} nm/fs", self.wave.t, packet.energy, TAU / k0, v),
            third,
            format!(
                "norm error {:+.0e}   electrons detected {}   shot {}",
                self.wave.norm() + self.wave.total_absorbed() - 1.0,
                self.detected,
                self.shots + 1
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A 1D wave of `n` points with the given potential (a function of x) and no absorber.
    fn wave_1d(n: usize, dx: f64, dt: f64, potential: impl Fn(f64) -> f64) -> Wave {
        let v = (0..n).map(|i| potential(coordinate(i, n, dx))).collect();
        Wave::new(n, 1, dx, dt, v, vec![0.0; n])
    }

    fn free_packet_1d(x0: f64, sigma: f64, energy: f64) -> Packet {
        Packet { x0, y0: 0.0, sigma_x: sigma, sigma_y: sigma, energy }
    }

    /// The barrier test: 0.5 eV on `points` grid points from x = 0, with the packet ready.
    fn barrier_wave(n: usize, dx: f64, dt: f64, points: usize) -> Wave {
        let mut potential = vec![0.0; n];
        for ix in n / 2..n / 2 + points {
            potential[ix] = BARRIER_HEIGHT;
        }
        let mut wave = Wave::new(n, 1, dx, dt, potential, vec![0.0; n]);
        wave.set_packet(&free_packet_1d(-30.0, 4.0, 0.375));
        wave
    }

    #[test]
    fn fft_matches_direct_sum() {
        let n = 16;
        let data: Vec<Complex> = (0..n).map(|j| Complex { re: (j as f64 * 0.7).sin() + 0.3 * j as f64, im: (j as f64 * 1.3).cos() }).collect();
        let fft = Fft::new(n);
        let mut transformed = data.clone();
        fft.transform(&mut transformed, false);
        for k in 0..n {
            let mut direct = Complex::ZERO;
            for (j, value) in data.iter().enumerate() {
                direct = direct + *value * Complex::from_phase(-TAU * (j * k) as f64 / n as f64);
            }
            assert!((direct - transformed[k]).norm_sqr().sqrt() < 1e-12, "bin {k}");
        }
        fft.transform(&mut transformed, true);
        for (a, b) in data.iter().zip(&transformed) {
            assert!((*a - *b).norm_sqr().sqrt() < 1e-12);
        }
    }

    #[test]
    fn free_packet_follows_exact_solution() {
        let mut wave = wave_1d(1024, DX, 0.05, |_| 0.0);
        let packet = free_packet_1d(-20.0, 1.0, 1.5);
        wave.set_packet(&packet);
        for _ in 0..800 {
            wave.step();
        }
        let t = 800.0 * 0.05;
        let v = HBAR * packet.k0() / M;
        let sigma = packet.sigma_x * (1.0 + (HBAR * t / (2.0 * M * packet.sigma_x.powi(2))).powi(2)).sqrt();
        let (mean, width, _, _) = wave.moments();
        assert!((mean - 9.055698).abs() < 1e-5 && (mean - (-20.0 + v * t)).abs() < 1e-9, "mean {mean}");
        assert!((width - 2.522074).abs() < 1e-5 && (width - sigma).abs() < 1e-9, "sigma {width}");
        // The density is the Gaussian of that centre and width.
        let peak = 1.0 / (TAU * sigma * sigma).sqrt();
        let worst = (0..1024)
            .map(|i| {
                let x = wave.x(i) - mean;
                let exact = peak * (-x * x / (2.0 * sigma * sigma)).exp();
                (wave.psi[i].norm_sqr() - exact).abs() / peak
            })
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "density off by {worst} of the peak");
        assert!((wave.norm() - 1.0).abs() < 1e-12);
        let expected_energy = 1.5 + HBAR * HBAR / (8.0 * M * packet.sigma_x.powi(2));
        assert!((wave.energy() - expected_energy).abs() < 1e-9, "energy {} expected {expected_energy}", wave.energy());
        assert!((expected_energy - 1.509525).abs() < 1e-6);
    }

    #[test]
    fn free_packet_spreads_in_y_too() {
        let (n, packet) = (256, Packet { x0: -8.0, y0: 0.0, sigma_x: 1.0, sigma_y: 1.5, energy: 1.5 });
        let mut wave = Wave::new(n, n, DX, 0.05, vec![0.0; n * n], vec![0.0; n * n]);
        wave.set_packet(&packet);
        for _ in 0..300 {
            wave.step();
        }
        let (mx, sx, my, sy) = wave.moments();
        let sigma_y = 1.5 * (1.0 + (HBAR * 15.0 / (2.0 * M * 1.5f64.powi(2))).powi(2)).sqrt();
        assert!((mx - 2.895887).abs() < 1e-5, "<x> {mx}");
        assert!(my.abs() < 1e-9, "<y> {my}");
        assert!((sx - 1.324338).abs() < 1e-5, "sigma x {sx}");
        assert!((sy - sigma_y).abs() < 1e-9 && (sy - 1.607810).abs() < 1e-5, "sigma y {sy}");
    }

    #[test]
    fn rectangular_barrier_transmission_matches_the_closed_form() {
        let packet = free_packet_1d(-30.0, 4.0, 0.375);
        let exact = packet_transmission(&packet, |e| rectangular_barrier_transmission(e, 0.5, 1.0));
        assert!((exact - 0.081239).abs() < 1e-5, "closed form {exact}");
        assert!((rectangular_barrier_transmission(0.375, 0.5, 1.0) - 0.077998).abs() < 1e-5);

        let mut wave = barrier_wave(2048, DX, 0.1, 8);
        for _ in 0..2000 {
            wave.step();
        }
        let transmitted = wave.probability_where(|ix| ix >= 1032);
        assert!((transmitted / exact - 1.0).abs() < 0.03, "T = {transmitted}, exact {exact}");
        let reflected = wave.probability_where(|ix| ix < 1032);
        assert!((reflected + transmitted - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rectangular_barrier_converges() {
        let exact = packet_transmission(&free_packet_1d(-30.0, 4.0, 0.375), |e| rectangular_barrier_transmission(e, 0.5, 1.0));
        let mut wave = barrier_wave(4096, 0.0625, 0.025, 16);
        for _ in 0..8000 {
            wave.step();
        }
        let transmitted = wave.probability_where(|ix| ix >= 2048 + 16);
        assert!((transmitted / exact - 1.0).abs() < 0.008, "T = {transmitted}, exact {exact}");
    }

    #[test]
    fn smooth_barrier_transmission() {
        // Eckart barrier V0 / cosh^2(x/w): T(k) = sinh^2(pi k w) / (sinh^2(pi k w) + cosh^2(pi/2 sqrt(B - 1))).
        let (v0, w) = (0.5, 0.5);
        let b = 8.0 * M * v0 * w * w / (HBAR * HBAR);
        let packet = free_packet_1d(-30.0, 4.0, 0.375);
        let exact = packet_transmission(&packet, |e| {
            let k = (2.0 * M * e).sqrt() / HBAR;
            let s = (PI * k * w).sinh().powi(2);
            s / (s + (PI / 2.0 * (b - 1.0).sqrt()).cosh().powi(2))
        });
        assert!((exact - 0.2598630).abs() < 1e-5, "closed form {exact}");
        let run = |dt: f64, steps: usize| {
            let mut wave = wave_1d(2048, DX, dt, |x| v0 / (x / w).cosh().powi(2));
            wave.set_packet(&packet);
            for _ in 0..steps {
                wave.step();
            }
            let transmitted = wave.probability_where(|ix| ix >= 1024);
            (transmitted / exact - 1.0).abs()
        };
        let (coarse, fine) = (run(0.1, 2000), run(0.05, 4000));
        assert!(coarse < 5e-4, "error {coarse} at dt 0.1");
        assert!(fine < 1.3e-4, "error {fine} at dt 0.05");
        let ratio = coarse / fine;
        assert!(ratio > 3.5 && ratio < 4.5, "error ratio {ratio}");
    }

    #[test]
    fn energy_is_conserved() {
        let mut wave = barrier_wave(2048, DX, 0.1, 8);
        let start = wave.energy();
        assert!((start - (0.375 + HBAR * HBAR / (8.0 * M * 16.0))).abs() < 1e-9, "start energy {start}");
        let mut worst: f64 = 0.0;
        for step in 1..=2000 {
            wave.step();
            if step % 20 == 0 {
                worst = worst.max((wave.energy() / start - 1.0).abs());
            }
        }
        assert!(worst < 3e-4, "energy changed by {worst}");
        assert!((wave.energy() / start - 1.0).abs() < 1e-10, "energy changed by {} in the end", wave.energy() / start - 1.0);
    }

    #[test]
    fn coherent_state_returns() {
        let omega = TAU / 20.0;
        let sigma = (HBAR / (2.0 * M * omega)).sqrt();
        let mut wave = wave_1d(256, DX, 0.05, |x| 0.5 * M * omega * omega * x * x);
        wave.set_packet(&free_packet_1d(4.0, sigma, 0.0));
        let start = wave.psi.clone();
        let start_energy = wave.energy();
        assert!((start_energy - (0.5 * M * omega * omega * 16.0 + 0.5 * HBAR * omega)).abs() < 1e-4, "start energy {start_energy}");
        for step in 1..=400 {
            wave.step();
            let (mean, width, _, _) = wave.moments();
            match step {
                100 => assert!(mean.abs() < 1e-3, "<x> at a quarter period {mean}"),
                200 => assert!((mean + 4.0).abs() < 1e-3, "<x> at half a period {mean}"),
                400 => assert!((mean - 4.0).abs() < 1e-3, "<x> after a period {mean}"),
                _ => {}
            }
            // The width breathes by 1.3e-5 at the quarter periods (the splitting error) and is
            // exact again at the half and full period.
            assert!((width - sigma).abs() < 5e-5, "width {width} at step {step}");
            if step % 200 == 0 {
                assert!((width - sigma).abs() < 1e-5, "width {width} at step {step}");
            }
        }
        // After a period psi returns as -psi (the zero-point phase).
        let overlap = wave.psi.iter().zip(&start).fold(Complex::ZERO, |sum, (now, then)| sum + then.conj() * *now).scale(DX);
        assert!(1.0 - overlap.norm_sqr() < 1e-6 && overlap.re < -0.9999, "overlap {overlap:?}");
        assert!((wave.energy() / start_energy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn absorber_swallows_the_packet() {
        for (energy, sigma, x0, steps, bound) in [(1.5, 2.0, -10.0, 2478, 1e-6), (0.375, 4.0, -8.0, 4846, 5e-5)] {
            let mut wave = Wave::new(512, 1, DX, 0.05, vec![0.0; 512], absorber(512, 1, DX));
            wave.set_packet(&free_packet_1d(x0, sigma, energy));
            for _ in 0..steps {
                wave.step();
            }
            assert!(wave.norm() < bound, "{energy} eV: {} left", wave.norm());
            assert!((wave.norm() + wave.total_absorbed() - 1.0).abs() < 1e-9, "probability not accounted for");
        }
    }

    #[test]
    fn barrier_scene_matches_one_dimension() {
        let mut sim = WavePacket::with_scene(Scene::barrier());
        for _ in 0..1200 {
            sim.step(0.1);
        }
        let (reflected, transmitted) = sim.split();
        assert!((transmitted - 0.07960).abs() < 2e-4, "transmitted {transmitted}");
        assert!((reflected + transmitted - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pick_row_follows_the_weights() {
        let weights = [0.5, -0.2, 0.0, 1.5];
        assert_eq!(pick_row(&weights, 0.2), 0);
        assert_eq!(pick_row(&weights, 0.5), 0);
        assert_eq!(pick_row(&weights, 0.51), 3);
        assert_eq!(pick_row(&weights, 2.0), 3);
    }

    #[test]
    fn single_electrons_arrive_one_by_one_where_the_probability_went() {
        let mut sim = WavePacket::with_scene_and_mode(Scene::double_slit(), true, 10.0);
        sim.screen[100] = 0.25;
        sim.screen[200] = 0.75;
        sim.screen[150] = -0.1; // backflow never makes a dot

        // Nothing before the first wave has passed.
        sim.fire_single_electrons(100.0);
        assert_eq!(sim.detected, 0);

        sim.shots = 1;
        macroquad::rand::srand(3);
        sim.fire_single_electrons(500.0);
        // Poisson: 5000 expected, standard deviation 71.
        assert!((sim.detected as f64 - 5000.0).abs() < 300.0, "{} electrons in 500 s at 10/s", sim.detected);
        let row_of = |dot: &Vec2| (dot.y as f64 / DX + (NY / 2) as f64).floor() as usize;
        let high = sim.dots.iter().filter(|dot| row_of(dot) == 200).count() as f64;
        assert!(sim.dots.iter().all(|dot| [100, 200].contains(&row_of(dot))));
        assert!((high / sim.dots.len() as f64 - 0.75).abs() < 0.03, "fraction on the bright row {}", high / sim.dots.len() as f64);
    }

    /// Position of the extremum near grid row `iy`, by a parabola through three points.
    fn refined(screen: &[f64], iy: usize) -> f64 {
        let (a, b, c) = (screen[iy - 1], screen[iy], screen[iy + 1]);
        let offset = 0.5 * (a - c) / (a - 2.0 * b + c);
        coordinate(iy, NY, DX) + offset * DX
    }

    #[test]
    fn double_slit_fringes() {
        let mut sim = WavePacket::with_scene(Scene::double_slit());
        let mut before_reset = sim.split();
        while sim.shots == 0 {
            before_reset = sim.split();
            sim.step(0.05);
        }
        let screen = &sim.screen;

        let most = screen.iter().cloned().fold(0.0, f64::max);
        for iy in 0..NY {
            assert!((screen[iy] - screen[NY - 1 - iy]).abs() < 1e-9 * most, "pattern not symmetric at row {iy}");
        }

        // Dark fringes: the lowest point in a window around each expected position.
        let lowest_near = |y: f64| {
            let range = (0..NY - 1).skip(1).filter(|&iy| (coordinate(iy, NY, DX) - y).abs() < 0.8);
            range.min_by(|&a, &b| screen[a].partial_cmp(&screen[b]).unwrap()).unwrap()
        };
        let (first, second) = (lowest_near(2.35), lowest_near(7.15));
        let (first_y, second_y) = (refined(screen, first), refined(screen, second));
        let centre = (NY / 2 - 1..NY / 2 + 1).map(|iy| screen[iy]).fold(0.0, f64::max);
        assert!(screen[first] < 0.02 * centre, "first dark fringe is {} of the centre", screen[first] / centre);

        let screen_x = sim.wave.x(sim.screen_ix);
        let lambda = TAU / sim.scene.packet.k0();
        let l = screen_x - WALL_X.1;
        let with_effective_separation = expected_dark_fringes(&sim.scene, screen_x);
        for (measured, expected) in [(first_y, with_effective_separation[0]), (second_y, with_effective_separation[1])] {
            assert!((measured / expected - 1.0).abs() < 0.003, "dark fringe at {measured}, two-source {expected}");
        }
        for (measured, n) in [(first_y, 0.5), (second_y, 1.5)] {
            let plain = dark_fringe(n, 8.0, l, lambda);
            assert!((measured / plain - 1.0).abs() < 0.01, "dark fringe at {measured}, plain d {plain}");
        }
        assert!((with_effective_separation[0] - 2.3471).abs() < 2e-3 && (with_effective_separation[1] - 7.1543).abs() < 2e-3);

        let arrived: f64 = screen.iter().sum();
        assert!((arrived - 0.0769).abs() < 0.003, "reached the screen: {arrived}");
        let (reflected, through) = before_reset;
        assert!((through - 0.1445).abs() < 0.003, "through the slits {through}");
        assert!((reflected - 0.8555).abs() < 0.003, "reflected {reflected}");
    }
}
