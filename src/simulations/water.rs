use std::cell::OnceCell;
use std::f64::consts::PI;

use macroquad::prelude::*;
use macroquad::rand::RandGenerator;

use crate::simulation::Simulation;

// Units: SI (metres, seconds, kilograms). The tank is 4 m x 2 m with a lid, seen from the side.

const G: f64 = 9.81; // m/s^2
const RHO: f64 = 1000.0; // kg/m^3, water

const NX: usize = 128;
const NY: usize = 64;
/// Cell size, m.
const H: f64 = 4.0 / NX as f64;
/// Time step, s. The fastest water (about 6 m/s, in the dam break) crosses 0.4 of a cell per step.
const DT: f64 = 1.0 / 480.0;
/// Particles per cell side when the water is seeded (2 x 2 per cell).
const SEEDS_PER_SIDE: usize = 2;
/// Fraction of each particle's new velocity from FLIP (the change on the grid added to its own
/// velocity); the rest is PIC (the grid velocity itself), which damps the particle noise.
const FLIP: f64 = 0.97;
/// Layers of grid velocity extrapolated into the air around the water.
const EXTRAPOLATE_LAYERS: usize = 4;
const CG_TOLERANCE: f64 = 1e-6;
const CG_MAX_ITERATIONS: usize = 400;

const MAX_STEPS_PER_FRAME: u32 = 16;
const MIN_SPEED: f64 = 0.125;
const MAX_SPEED: f64 = 2.0;
const START_SPEED: f64 = 0.5;

// The dam break: a column of water held against the left wall, let go.
const DAM_WIDTH: f64 = 1.0;
const DAM_HEIGHT: f64 = 1.2;

// The sloshing scene: water 0.8 m deep with its surface tilted along the tank's first mode.
const SLOSH_DEPTH: f64 = 0.8;
const SLOSH_AMPLITUDE: f64 = 0.1;

/// Particles carrying the water's velocity: Fluid-Implicit-Particle (FLIP).
struct Particles {
    x: Vec<f64>,
    y: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
}

/// The water on a staggered (MAC) grid of NX x NY cells: u on the left face of each cell, v on
/// the bottom face, pressure at the centre. The tank's walls are the grid's edges. A cell with a
/// particle in it is water, any other is air (zero pressure: the free surface).
/// No macroquad types, so tests can build it without a window.
struct Tank {
    particles: Particles,
    /// `u[j * (NX + 1) + i]`, `v[j * NX + i]`.
    u: Vec<f64>,
    v: Vec<f64>,
    /// The grid velocities before the forces and pressure, for the FLIP update.
    u_before: Vec<f64>,
    v_before: Vec<f64>,
    water: Vec<bool>,
    /// dt p / rho at each cell centre, m^2/s, from the last pressure solve.
    q: Vec<f64>,
    cg_iterations: usize,
    t: f64,
}

/// Bilinear weights of a point (in units of cells, relative to the first sample) over the four
/// samples around it: their indices (clamped to the array) and weights.
fn stencil(fx: f64, fy: f64, width: usize, height: usize) -> [(usize, f64); 4] {
    let (i, j) = (fx.floor(), fy.floor());
    let (sx, sy) = (fx - i, fy - j);
    let clamp_i = |i: f64| (i.max(0.0) as usize).min(width - 1);
    let clamp_j = |j: f64| (j.max(0.0) as usize).min(height - 1);
    let index = |di: f64, dj: f64| clamp_j(j + dj) * width + clamp_i(i + di);
    [
        (index(0.0, 0.0), (1.0 - sx) * (1.0 - sy)),
        (index(1.0, 0.0), sx * (1.0 - sy)),
        (index(0.0, 1.0), (1.0 - sx) * sy),
        (index(1.0, 1.0), sx * sy),
    ]
}

/// Where in the u grid a point is (u sits at (i h, (j + 1/2) h)), and in the v grid.
fn u_stencil(x: f64, y: f64) -> [(usize, f64); 4] {
    stencil(x / H, y / H - 0.5, NX + 1, NY)
}

fn v_stencil(x: f64, y: f64) -> [(usize, f64); 4] {
    stencil(x / H - 0.5, y / H, NX, NY + 1)
}

fn sample(field: &[f64], weights: &[(usize, f64); 4]) -> f64 {
    weights.iter().map(|&(k, w)| w * field[k]).sum()
}

impl Tank {
    /// Water wherever `inside(x, y)` holds, seeded 2 x 2 per cell with a little jitter, at rest.
    fn new(inside: impl Fn(f64, f64) -> bool, seed: u64) -> Self {
        let rng = RandGenerator::new();
        rng.srand(seed);
        let mut particles = Particles { x: Vec::new(), y: Vec::new(), u: Vec::new(), v: Vec::new() };
        let spacing = H / SEEDS_PER_SIDE as f64;
        for row in 0..NY * SEEDS_PER_SIDE {
            for col in 0..NX * SEEDS_PER_SIDE {
                let x = (col as f64 + 0.5 + rng.gen_range(-0.3, 0.3)) * spacing;
                let y = (row as f64 + 0.5 + rng.gen_range(-0.3, 0.3)) * spacing;
                if inside(x, y) {
                    particles.x.push(x);
                    particles.y.push(y);
                    particles.u.push(0.0);
                    particles.v.push(0.0);
                }
            }
        }
        Self {
            particles,
            u: vec![0.0; (NX + 1) * NY],
            v: vec![0.0; NX * (NY + 1)],
            u_before: vec![0.0; (NX + 1) * NY],
            v_before: vec![0.0; NX * (NY + 1)],
            water: vec![false; NX * NY],
            q: vec![0.0; NX * NY],
            cg_iterations: 0,
            t: 0.0,
        }
    }

    fn step(&mut self, dt: f64) {
        self.particles_to_grid();
        self.u_before.clone_from(&self.u);
        self.v_before.clone_from(&self.v);
        for v in &mut self.v {
            *v -= G * dt;
        }
        self.close_walls();
        self.project();
        self.extrapolate();
        self.grid_to_particles();
        self.move_particles(dt);
        self.t += dt;
    }

    /// Each grid velocity becomes the weighted average of the particles near it; cells with a
    /// particle become water. Faces no particle reaches are filled from their neighbours.
    fn particles_to_grid(&mut self) {
        let p = &self.particles;
        let (mut u_weight, mut v_weight) = (vec![0.0; self.u.len()], vec![0.0; self.v.len()]);
        self.u.iter_mut().for_each(|u| *u = 0.0);
        self.v.iter_mut().for_each(|v| *v = 0.0);
        self.water.iter_mut().for_each(|w| *w = false);
        for n in 0..p.x.len() {
            for (k, w) in u_stencil(p.x[n], p.y[n]) {
                self.u[k] += w * p.u[n];
                u_weight[k] += w;
            }
            for (k, w) in v_stencil(p.x[n], p.y[n]) {
                self.v[k] += w * p.v[n];
                v_weight[k] += w;
            }
            let (i, j) = (((p.x[n] / H) as usize).min(NX - 1), ((p.y[n] / H) as usize).min(NY - 1));
            self.water[j * NX + i] = true;
        }
        for (u, w) in self.u.iter_mut().zip(&u_weight) {
            if *w > 0.0 {
                *u /= w;
            }
        }
        for (v, w) in self.v.iter_mut().zip(&v_weight) {
            if *w > 0.0 {
                *v /= w;
            }
        }
        self.extrapolate();
    }

    /// No flow through the walls.
    fn close_walls(&mut self) {
        for j in 0..NY {
            self.u[j * (NX + 1)] = 0.0;
            self.u[j * (NX + 1) + NX] = 0.0;
        }
        for i in 0..NX {
            self.v[i] = 0.0;
            self.v[NY * NX + i] = 0.0;
        }
    }

    fn is_water(&self, i: isize, j: isize) -> bool {
        i >= 0 && j >= 0 && (i as usize) < NX && (j as usize) < NY && self.water[j as usize * NX + i as usize]
    }

    fn is_wall(i: isize, j: isize) -> bool {
        i < 0 || j < 0 || i as usize >= NX || j as usize >= NY
    }

    /// The pressure that makes the water incompressible. With q = dt p / rho, each water cell
    /// needs sum over its open sides of (q_cell - q_neighbour) = -h^2 div u; the neighbour's q is
    /// 0 in air (the free surface) and walls drop out (their face velocity is already zero).
    /// Conjugate gradients with a modified incomplete Cholesky preconditioner (Bridson 2015).
    fn project(&mut self) {
        let n = NX * NY;
        let mut rhs = vec![0.0; n];
        for j in 0..NY {
            for i in 0..NX {
                if self.water[j * NX + i] {
                    let div = self.u[j * (NX + 1) + i + 1] - self.u[j * (NX + 1) + i] + self.v[(j + 1) * NX + i] - self.v[j * NX + i];
                    rhs[j * NX + i] = -div * H;
                }
            }
        }
        let matrix = Matrix::new(self);
        let (q, iterations) = matrix.solve(&rhs);
        self.cg_iterations = iterations;
        self.q = q;

        // Subtract the gradient on every face next to water.
        for j in 0..NY as isize {
            for i in 1..NX as isize {
                if self.is_water(i - 1, j) || self.is_water(i, j) {
                    let k = j as usize * (NX + 1) + i as usize;
                    self.u[k] -= (self.q_at(i, j) - self.q_at(i - 1, j)) / H;
                }
            }
        }
        for j in 1..NY as isize {
            for i in 0..NX as isize {
                if self.is_water(i, j - 1) || self.is_water(i, j) {
                    let k = j as usize * NX + i as usize;
                    self.v[k] -= (self.q_at(i, j) - self.q_at(i, j - 1)) / H;
                }
            }
        }
    }

    fn q_at(&self, i: isize, j: isize) -> f64 {
        if self.is_water(i, j) { self.q[j as usize * NX + i as usize] } else { 0.0 }
    }

    /// Faces next to water (and the walls) have a velocity; each pass gives the faces just beyond
    /// them the average of their known neighbours, so particles near the surface sample sensible
    /// values.
    fn extrapolate(&mut self) {
        let mut u_known: Vec<bool> = (0..self.u.len())
            .map(|k| {
                let (i, j) = ((k % (NX + 1)) as isize, (k / (NX + 1)) as isize);
                i == 0 || i == NX as isize || self.is_water(i - 1, j) || self.is_water(i, j)
            })
            .collect();
        let mut v_known: Vec<bool> = (0..self.v.len())
            .map(|k| {
                let (i, j) = ((k % NX) as isize, (k / NX) as isize);
                j == 0 || j == NY as isize || self.is_water(i, j - 1) || self.is_water(i, j)
            })
            .collect();
        for _ in 0..EXTRAPOLATE_LAYERS {
            extrapolate_layer(&mut self.u, &mut u_known, NX + 1, NY);
            extrapolate_layer(&mut self.v, &mut v_known, NX, NY + 1);
        }
        for (u, known) in self.u.iter_mut().zip(&u_known) {
            if !known {
                *u = 0.0;
            }
        }
        for (v, known) in self.v.iter_mut().zip(&v_known) {
            if !known {
                *v = 0.0;
            }
        }
        self.close_walls();
    }

    /// FLIP: each particle keeps its own velocity plus the change the grid made, blended with a
    /// little of the grid velocity itself.
    fn grid_to_particles(&mut self) {
        let p = &mut self.particles;
        for n in 0..p.x.len() {
            let (us, vs) = (u_stencil(p.x[n], p.y[n]), v_stencil(p.x[n], p.y[n]));
            let (u_new, v_new) = (sample(&self.u, &us), sample(&self.v, &vs));
            let (du, dv) = (u_new - sample(&self.u_before, &us), v_new - sample(&self.v_before, &vs));
            p.u[n] = FLIP * (p.u[n] + du) + (1.0 - FLIP) * u_new;
            p.v[n] = FLIP * (p.v[n] + dv) + (1.0 - FLIP) * v_new;
        }
    }

    /// Midpoint rule through the grid velocity, kept inside the tank.
    fn move_particles(&mut self, dt: f64) {
        let margin = 0.01 * H;
        let (width, height) = (NX as f64 * H - margin, NY as f64 * H - margin);
        let p = &mut self.particles;
        for n in 0..p.x.len() {
            let velocity = |x: f64, y: f64| (sample(&self.u, &u_stencil(x, y)), sample(&self.v, &v_stencil(x, y)));
            let (u, v) = velocity(p.x[n], p.y[n]);
            let (u, v) = velocity(p.x[n] + 0.5 * dt * u, p.y[n] + 0.5 * dt * v);
            p.x[n] = (p.x[n] + dt * u).clamp(margin, width);
            p.y[n] = (p.y[n] + dt * v).clamp(margin, height);
        }
    }

    /// Pressure at the centre of a cell, Pa (0 in air).
    fn pressure(&self, i: usize, j: usize, dt: f64) -> f64 {
        RHO * self.q_at(i as isize, j as isize) / dt
    }

    /// Mean x of the water, m.
    fn centre_x(&self) -> f64 {
        self.particles.x.iter().sum::<f64>() / self.particles.x.len() as f64
    }

    fn max_speed(&self) -> f64 {
        let p = &self.particles;
        p.u.iter().zip(&p.v).map(|(u, v)| u.hypot(*v)).fold(0.0, f64::max)
    }

    /// The furthest any water has reached to the right, m: the front of the dam break.
    fn front(&self) -> f64 {
        // The rightmost water cell on the bottom row, so a few stray drops don't count.
        (0..NX).rev().find(|&i| self.water[i]).map_or(0.0, |i| (i + 1) as f64 * H)
    }
}

fn extrapolate_layer(field: &mut [f64], known: &mut [bool], width: usize, height: usize) {
    let mut newly = Vec::new();
    for j in 0..height {
        for i in 0..width {
            let k = j * width + i;
            if known[k] {
                continue;
            }
            let (mut sum, mut count) = (0.0, 0);
            let neighbours = [(i.wrapping_sub(1), j), (i + 1, j), (i, j.wrapping_sub(1)), (i, j + 1)];
            for (ni, nj) in neighbours {
                if ni < width && nj < height && known[nj * width + ni] {
                    sum += field[nj * width + ni];
                    count += 1;
                }
            }
            if count > 0 {
                newly.push((k, sum / count as f64));
            }
        }
    }
    for (k, value) in newly {
        field[k] = value;
        known[k] = true;
    }
}

/// The pressure matrix over the water cells: the diagonal counts the open (non-wall) sides, and
/// -1 links neighbouring water cells.
struct Matrix {
    water: Vec<bool>,
    diagonal: Vec<f64>,
    /// Link to the cell on the right, and to the cell above (0 or -1).
    plus_i: Vec<f64>,
    plus_j: Vec<f64>,
    /// The MIC(0) preconditioner: 1 / sqrt of the diagonal of its factor.
    precon: Vec<f64>,
}

impl Matrix {
    fn new(tank: &Tank) -> Self {
        let n = NX * NY;
        let (mut diagonal, mut plus_i, mut plus_j) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        for j in 0..NY as isize {
            for i in 0..NX as isize {
                if !tank.is_water(i, j) {
                    continue;
                }
                let k = j as usize * NX + i as usize;
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if !Tank::is_wall(i + di, j + dj) {
                        diagonal[k] += 1.0;
                    }
                }
                if tank.is_water(i + 1, j) {
                    plus_i[k] = -1.0;
                }
                if tank.is_water(i, j + 1) {
                    plus_j[k] = -1.0;
                }
            }
        }
        // Modified incomplete Cholesky with the usual tuning (tau 0.97, safety 0.25).
        let (tau, safety) = (0.97, 0.25);
        let mut precon = vec![0.0f64; n];
        for j in 0..NY {
            for i in 0..NX {
                let k = j * NX + i;
                if !tank.water[k] {
                    continue;
                }
                let (left, below) = (if i > 0 { k - 1 } else { k }, if j > 0 { k - NX } else { k });
                let (a_left, a_below) = (if i > 0 { plus_i[left] } else { 0.0 }, if j > 0 { plus_j[below] } else { 0.0 });
                let mut e = diagonal[k] - (a_left * precon[left]).powi(2) - (a_below * precon[below]).powi(2)
                    - tau * (a_left * plus_j[left] * precon[left].powi(2) + a_below * plus_i[below] * precon[below].powi(2));
                if e < safety * diagonal[k] {
                    e = diagonal[k];
                }
                precon[k] = 1.0 / e.sqrt();
            }
        }
        Self { water: tank.water.clone(), diagonal, plus_i, plus_j, precon }
    }

    fn apply(&self, x: &[f64], out: &mut [f64]) {
        for j in 0..NY {
            for i in 0..NX {
                let k = j * NX + i;
                if !self.water[k] {
                    out[k] = 0.0;
                    continue;
                }
                let mut sum = self.diagonal[k] * x[k];
                if i + 1 < NX {
                    sum += self.plus_i[k] * x[k + 1];
                }
                if i > 0 {
                    sum += self.plus_i[k - 1] * x[k - 1];
                }
                if j + 1 < NY {
                    sum += self.plus_j[k] * x[k + NX];
                }
                if j > 0 {
                    sum += self.plus_j[k - NX] * x[k - NX];
                }
                out[k] = sum;
            }
        }
    }

    /// z = M^-1 r: forward then backward substitution with the incomplete factor.
    fn precondition(&self, r: &[f64], z: &mut [f64]) {
        let mut q = vec![0.0; r.len()];
        for j in 0..NY {
            for i in 0..NX {
                let k = j * NX + i;
                if !self.water[k] {
                    continue;
                }
                let mut t = r[k];
                if i > 0 {
                    t -= self.plus_i[k - 1] * self.precon[k - 1] * q[k - 1];
                }
                if j > 0 {
                    t -= self.plus_j[k - NX] * self.precon[k - NX] * q[k - NX];
                }
                q[k] = t * self.precon[k];
            }
        }
        for j in (0..NY).rev() {
            for i in (0..NX).rev() {
                let k = j * NX + i;
                if !self.water[k] {
                    z[k] = 0.0;
                    continue;
                }
                let mut t = q[k];
                if i + 1 < NX {
                    t -= self.plus_i[k] * self.precon[k] * z[k + 1];
                }
                if j + 1 < NY {
                    t -= self.plus_j[k] * self.precon[k] * z[k + NX];
                }
                z[k] = t * self.precon[k];
            }
        }
    }

    /// Preconditioned conjugate gradients from zero, until the largest residual is below the
    /// tolerance (relative to the largest right-hand side). Returns the solution and iterations.
    fn solve(&self, rhs: &[f64]) -> (Vec<f64>, usize) {
        let n = rhs.len();
        let mut x = vec![0.0; n];
        let mut r = rhs.to_vec();
        let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if scale == 0.0 {
            return (x, 0);
        }
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        let mut z = vec![0.0; n];
        self.precondition(&r, &mut z);
        let mut s = z.clone();
        let mut sigma = dot(&z, &r);
        let mut a_s = vec![0.0; n];
        for iteration in 1..=CG_MAX_ITERATIONS {
            self.apply(&s, &mut a_s);
            let alpha = sigma / dot(&s, &a_s);
            for k in 0..n {
                x[k] += alpha * s[k];
                r[k] -= alpha * a_s[k];
            }
            if r.iter().fold(0.0f64, |m, v| m.max(v.abs())) <= CG_TOLERANCE * scale {
                return (x, iteration);
            }
            self.precondition(&r, &mut z);
            let sigma_new = dot(&z, &r);
            let beta = sigma_new / sigma;
            for k in 0..n {
                s[k] = z[k] + beta * s[k];
            }
            sigma = sigma_new;
        }
        (x, CG_MAX_ITERATIONS)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Scene {
    DamBreak,
    Slosh,
    Drop,
}

/// Angular frequency of the tank's first sloshing mode in linear theory: a standing wave of
/// wave number k = pi / L on water of depth d, omega^2 = g k tanh(k d).
fn slosh_frequency(length: f64, depth: f64) -> f64 {
    let k = PI / length;
    (G * k * (k * depth).tanh()).sqrt()
}

impl Scene {
    fn tank(self) -> Tank {
        let length = NX as f64 * H;
        match self {
            Scene::DamBreak => Tank::new(|x, y| x < DAM_WIDTH && y < DAM_HEIGHT, 1),
            Scene::Slosh => Tank::new(|x, y| y < SLOSH_DEPTH + SLOSH_AMPLITUDE * (PI * x / length).cos(), 2),
            // A ball of water 0.6 m across dropped from 1.2 m above a pool 0.5 m deep.
            Scene::Drop => Tank::new(|x, y| y < 0.5 || (x - 2.0).powi(2) + (y - 1.6).powi(2) < 0.09, 3),
        }
    }
}

/// Water in a closed tank, from the incompressible Navier-Stokes equations with gravity and a free
/// surface, in three scenes (keys 1-3): a dam break (a column of water collapses and surges
/// along the floor, hits the far wall and plunges back), a standing wave sloshing from side to
/// side (its period is shown against linear theory), and a ball of water dropped into a pool.
/// FLIP (Fluid-Implicit-Particle; Brackbill and Ruppel 1986, Zhu and Bridson 2005): particles
/// carry the water and its velocity, a 128 x 64 staggered grid enforces incompressibility with a
/// pressure that is zero at the surface. Colour is speed.
/// Not modelled: viscosity (water's is far too small to matter here: Re ~ 1e6, and the grid's
/// own numerical diffusion is larger), surface tension, air (it has no pressure, so a trapped
/// pocket collapses at once), turbulence below the 3 cm grid, the third dimension, and exact volume (FLIP keeps
/// the particles, but they can bunch or spread by a few per cent).
pub struct Water {
    scene: Scene,
    tank: Tank,
    speed: f64,
    steps_this_frame: u32,
    capped: bool,
    /// Sloshing: times the water's centre crossed the middle going right, s.
    crossings: Vec<f64>,
    /// Dam break: when the front reached the far wall, s.
    front_at_wall: Option<f64>,
    previous_offset: f64,
    texture: OnceCell<Texture2D>,
}

impl Water {
    pub fn new() -> Self {
        Self::with_scene(Scene::DamBreak)
    }

    fn with_scene(scene: Scene) -> Self {
        let tank = scene.tank();
        let previous_offset = tank.centre_x() - NX as f64 * H / 2.0;
        Self { scene, tank, speed: START_SPEED, steps_this_frame: 0, capped: false, crossings: Vec::new(), front_at_wall: None, previous_offset, texture: OnceCell::new() }
    }

    /// Measured sloshing period from the crossings so far, s.
    fn slosh_period(&self) -> Option<f64> {
        let n = self.crossings.len();
        (n >= 2).then(|| (self.crossings[n - 1] - self.crossings[0]) / (n - 1) as f64)
    }
}

impl Simulation for Water {
    fn step(&mut self, dt: f64) {
        if self.steps_this_frame >= MAX_STEPS_PER_FRAME {
            self.capped = true;
            return;
        }
        self.steps_this_frame += 1;
        self.tank.step(dt);

        // The centre of the water swings with the first mode: time its upward crossings.
        let offset = self.tank.centre_x() - NX as f64 * H / 2.0;
        if self.previous_offset < 0.0 && offset >= 0.0 {
            let t = self.tank.t - dt * offset / (offset - self.previous_offset);
            self.crossings.push(t);
        }
        self.previous_offset = offset;
        if self.front_at_wall.is_none() && self.tank.front() >= NX as f64 * H - 1e-9 {
            self.front_at_wall = Some(self.tank.t);
        }
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
        // The 4 m x 2 m tank below the HUD lines, at 3:2.
        Rect::new(-0.1, -0.1, 4.2, 2.8)
    }

    fn handle_input(&mut self) {
        self.steps_this_frame = 0;
        self.capped = false;
        for (key, scene) in [(KeyCode::Key1, Scene::DamBreak), (KeyCode::Key2, Scene::Slosh), (KeyCode::Key3, Scene::Drop)] {
            if is_key_pressed(key) {
                let speed = self.speed;
                *self = Self::with_scene(scene);
                self.speed = speed;
            }
        }
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
    }

    fn draw(&self) {
        // Particles splatted into a texture four times finer than the grid as 5 x 5 squares, which
        // overlap their neighbours (2 pixels apart), so the water looks solid. Colour is the mean
        // speed of the particles covering a pixel: deep blue at rest, white at 5 m/s.
        let (w, h) = (4 * NX, 4 * NY);
        let (mut count, mut speed) = (vec![0u16; w * h], vec![0.0f32; w * h]);
        let p = &self.tank.particles;
        let scale = 4.0 / H;
        for n in 0..p.x.len() {
            let s = p.u[n].hypot(p.v[n]) as f32;
            let (i, j) = ((p.x[n] * scale) as isize, (p.y[n] * scale) as isize);
            for pj in (j - 2).max(0)..(j + 3).min(h as isize) {
                for pi in (i - 2).max(0)..(i + 3).min(w as isize) {
                    let k = pj as usize * w + pi as usize;
                    count[k] += 1;
                    speed[k] += s;
                }
            }
        }
        let mut pixels = Vec::with_capacity(4 * w * h);
        for k in 0..w * h {
            if count[k] == 0 {
                pixels.extend_from_slice(&[12, 14, 22, 255]);
            } else {
                let s = (speed[k] / count[k] as f32 / 5.0).min(1.0);
                pixels.extend_from_slice(&[(30.0 + 225.0 * s) as u8, (90.0 + 165.0 * s) as u8, (200.0 + 55.0 * s) as u8, 255]);
            }
        }
        let texture = self.texture.get_or_init(|| {
            let texture = Texture2D::from_rgba8(w as u16, h as u16, &pixels);
            texture.set_filter(FilterMode::Linear);
            texture
        });
        texture.update_from_bytes(w as u32, h as u32, &pixels);
        let (width, height) = ((NX as f64 * H) as f32, (NY as f64 * H) as f32);
        draw_texture_ex(texture, 0.0, 0.0, WHITE, DrawTextureParams { dest_size: Some(vec2(width, height)), ..Default::default() });
        // The tank: floor, walls and lid.
        let wall = 0.03;
        draw_rectangle(-wall, -wall, width + 2.0 * wall, wall, GRAY);
        draw_rectangle(-wall, height, width + 2.0 * wall, wall, GRAY);
        draw_rectangle(-wall, 0.0, wall, height, GRAY);
        draw_rectangle(width, 0.0, wall, height, GRAY);
    }

    fn hud(&self) -> Vec<String> {
        let tank = &self.tank;
        let mut lines = vec![
            format!(
                "[1] dam break  [2] sloshing  [3] drop   t = {:.2} s   speed {}x (UP/DOWN){}",
                tank.t,
                self.speed,
                if self.capped { " - capped" } else { "" }
            ),
            format!(
                "{} particles   fastest {:.2} m/s   pressure solve {} iterations",
                tank.particles.x.len(),
                tank.max_speed(),
                tank.cg_iterations
            ),
        ];
        match self.scene {
            Scene::DamBreak => {
                // Ritter's (1892) dam-break solution for shallow water: the front runs at
                // 2 sqrt(g h0) from the start. It ignores the vertical motion (the column first
                // has to fall) and the floor's friction, so real fronts lag it.
                let length = NX as f64 * H;
                let front_speed = 2.0 * (G * DAM_HEIGHT).sqrt();
                let ritter_at_wall = (length - DAM_WIDTH) / front_speed;
                lines.push(match self.front_at_wall {
                    None => format!("front at {:.2} m   Ritter's ideal front {:.2} m", tank.front(), (DAM_WIDTH + front_speed * tank.t).min(length)),
                    Some(t) => format!("front hit the far wall at {:.2} s   Ritter's ideal {:.2} s", t, ritter_at_wall),
                });
            }
            Scene::Slosh => {
                let theory = 2.0 * PI / slosh_frequency(NX as f64 * H, SLOSH_DEPTH);
                let measured = self.slosh_period().map_or("-".to_string(), |p| format!("{p:.3} s"));
                lines.push(format!("sloshing period {}   linear theory {:.3} s", measured, theory));
            }
            Scene::Drop => {
                lines.push(format!("pressure at the bottom centre {:.1} kPa", tank.pressure(NX / 2, 0, DT) / 1000.0));
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_water_stays_still_with_hydrostatic_pressure() {
        let depth = 0.5;
        let mut tank = Tank::new(|_, y| y < depth, 4);
        for _ in 0..480 {
            tank.step(DT);
        }
        assert!(tank.max_speed() < 0.02, "still water moves at {} m/s", tank.max_speed());
        // The bottom cells' centres are h/2 above the floor, so rho g (d - h/2) deep; but the discrete surface (p = 0) is the first air cells' centres,
        // h/2 above the water. The two cancel: rho g d.
        let expected = RHO * G * depth;
        for i in [10, NX / 2, NX - 10] {
            let p = tank.pressure(i, 0, DT);
            assert!((p / expected - 1.0).abs() < 0.01, "pressure {p} Pa at column {i}, expected {expected}");
        }
    }

    #[test]
    fn pressure_solve_makes_the_water_incompressible() {
        let mut tank = Scene::DamBreak.tank();
        for _ in 0..200 {
            tank.step(DT);
        }
        // Divergence in water cells, relative to the speeds involved.
        let mut worst: f64 = 0.0;
        for j in 0..NY {
            for i in 0..NX {
                if tank.water[j * NX + i] {
                    let div = tank.u[j * (NX + 1) + i + 1] - tank.u[j * (NX + 1) + i] + tank.v[(j + 1) * NX + i] - tank.v[j * NX + i];
                    worst = worst.max(div.abs());
                }
            }
        }
        assert!(worst < 1e-4 * tank.max_speed(), "divergence {worst}");
    }

    #[test]
    fn sloshing_period_matches_linear_theory() {
        let mut water = Water::with_scene(Scene::Slosh);
        while water.crossings.len() < 4 {
            water.steps_this_frame = 0;
            water.step(DT);
        }
        let measured = water.slosh_period().unwrap();
        let theory = 2.0 * PI / slosh_frequency(NX as f64 * H, SLOSH_DEPTH);
        assert!((measured / theory - 1.0).abs() < 0.03, "period {measured} s, theory {theory} s");
    }

    #[test]
    fn the_dam_break_front_starts_at_the_column() {
        let mut tank = Scene::DamBreak.tank();
        tank.particles_to_grid();
        assert!((tank.front() - DAM_WIDTH).abs() < 1e-9, "front {}", tank.front());
        // After 0.3 s the front has moved, but not as far as Ritter's ideal front.
        while tank.t < 0.3 {
            tank.step(DT);
        }
        let ritter = DAM_WIDTH + 2.0 * (G * DAM_HEIGHT).sqrt() * tank.t;
        assert!(tank.front() > DAM_WIDTH + 0.3 && tank.front() < ritter, "front at {} m", tank.front());
    }
}

