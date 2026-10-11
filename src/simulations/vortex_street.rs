use std::cell::OnceCell;

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: the solver works in lattice units (cell size 1, time step 1, density 1, sound speed
// squared 1/3, so pressure is density / 3). The HUD converts to SI: metres and seconds, for water.

/// D2Q9 lattice: velocity vectors, weights and the index of the opposite direction.
const C: [(i32, i32); 9] = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)];
const W: [f64; 9] = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];
const OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];
/// The four pairs of opposite directions handled by the TRT collision.
const PAIRS: [(usize, usize); 4] = [(1, 3), (2, 4), (5, 7), (6, 8)];

/// Cells across the cylinder. N = 10 is unstable at Re 100 (tau = 0.52); 30 does not fit a frame.
const CELLS_PER_DIAMETER: usize = 20;
/// Peak inlet speed in lattice units (Mach 0.17). The mean is 2/3 of it.
const PEAK_INLET_SPEED: f64 = 0.1;
const MEAN_INLET_SPEED: f64 = 2.0 / 3.0 * PEAK_INLET_SPEED;
/// Magic parameter of the two-relaxation-time collision, which sets where the wall sits.
const MAGIC: f64 = 0.25;

// The real case: water at 20 C in a channel past a 1 cm rod (Schaefer and Turek's geometry).
const VISCOSITY: f64 = 1.0034e-6; // m^2/s, from mu = 1.0016 mPa s and rho = 998.21 kg/m^3
const DIAMETER: f64 = 0.01; // m
const CHANNEL_LENGTH: f64 = 22.0 * DIAMETER;
const REYNOLDS_UNSTEADY: f64 = 100.0;
const REYNOLDS_STEADY: f64 = 20.0;

const MAX_STEPS_PER_FRAME: u32 = 12;
const MIN_SPEED: f64 = 0.25;
const MAX_SPEED: f64 = 16.0;

#[derive(Clone, Copy, PartialEq)]
enum Node {
    Fluid,
    Wall,
    Cylinder,
}

/// Equilibrium populations (incompressible variant, He and Luo 1997): `rho` is the density and
/// `(ux, uy)` the velocity, which is the momentum because the reference density is 1.
fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; 9] {
    let usq = ux * ux + uy * uy;
    let mut f = [0.0; 9];
    for q in 0..9 {
        let cu = C[q].0 as f64 * ux + C[q].1 as f64 * uy;
        f[q] = W[q] * (rho + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
    }
    f
}

/// A lattice Boltzmann solver: D2Q9, two-relaxation-time collision, bounce-back walls, Zou-He
/// open ends. No macroquad types, so tests can build it without a window.
struct Lattice {
    nx: usize,
    ny: usize,
    /// Populations arriving at each node, 9 per node: index `9 * (j * nx + i) + q`.
    f: Vec<f64>,
    /// The same after collision.
    f_post: Vec<f64>,
    nodes: Vec<Node>,
    omega_plus: f64,
    omega_minus: f64,
    /// Prescribed u_x for each row at the inlet; `None` means periodic in x (tests only).
    inlet: Option<Vec<f64>>,
    /// Force on the cylinder during the last step.
    force: (f64, f64),
}

impl Lattice {
    /// `nu` is the kinematic viscosity in lattice units. The fluid starts at equilibrium with
    /// density 1 and the inlet velocity (or at rest if `inlet` is `None`).
    fn new(nx: usize, ny: usize, nu: f64, nodes: Vec<Node>, inlet: Option<Vec<f64>>) -> Self {
        let tau = 3.0 * nu + 0.5;
        let tau_minus = 0.5 + MAGIC / (tau - 0.5);
        let mut f = vec![0.0; 9 * nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if nodes[k] == Node::Fluid {
                    let ux = inlet.as_ref().map_or(0.0, |u| u[j]);
                    f[9 * k..9 * k + 9].copy_from_slice(&equilibrium(1.0, ux, 0.0));
                }
            }
        }
        Self { nx, ny, f_post: f.clone(), f, nodes, omega_plus: 1.0 / tau, omega_minus: 1.0 / tau_minus, inlet, force: (0.0, 0.0) }
    }

    fn step(&mut self) {
        self.collide();
        self.stream();
        if self.inlet.is_some() {
            self.apply_open_ends();
        }
    }

    /// Collision (TRT) at every fluid node. The part of each opposite pair that is even under
    /// reversing the direction relaxes at `omega_plus` (this sets the viscosity), the odd part at
    /// `omega_minus`.
    fn collide(&mut self) {
        for k in 0..self.nodes.len() {
            if self.nodes[k] != Node::Fluid {
                continue;
            }
            let f = &self.f[9 * k..9 * k + 9];
            let rho: f64 = f.iter().sum();
            let ux: f64 = (0..9).map(|q| f[q] * C[q].0 as f64).sum();
            let uy: f64 = (0..9).map(|q| f[q] * C[q].1 as f64).sum();
            let usq = ux * ux + uy * uy;
            let out = &mut self.f_post[9 * k..9 * k + 9];
            out[0] = f[0] - self.omega_plus * (f[0] - 4.0 / 9.0 * (rho - 1.5 * usq));
            for (q, opposite) in PAIRS {
                let cu = C[q].0 as f64 * ux + C[q].1 as f64 * uy;
                let even = self.omega_plus * ((f[q] + f[opposite]) / 2.0 - W[q] * (rho + 4.5 * cu * cu - 1.5 * usq));
                let odd = self.omega_minus * ((f[q] - f[opposite]) / 2.0 - 3.0 * W[q] * cu);
                out[q] = f[q] - even - odd;
                out[opposite] = f[opposite] - even + odd;
            }
        }
    }

    /// Streaming in pull form: each fluid node takes population q from the node it came from, one
    /// step back along c_q (wrapping around the grid). If that node is solid, the population that
    /// left towards it comes straight back (bounce-back), which puts the wall halfway between the
    /// two nodes. For a cylinder node the momentum exchanged is the force on the cylinder.
    fn stream(&mut self) {
        let (nx, ny) = (self.nx as i32, self.ny as i32);
        let mut force = (0.0, 0.0);
        for j in 0..ny {
            for i in 0..nx {
                let k = (j * nx + i) as usize;
                if self.nodes[k] != Node::Fluid {
                    continue;
                }
                for q in 0..9 {
                    let (si, sj) = ((i - C[q].0).rem_euclid(nx), (j - C[q].1).rem_euclid(ny));
                    let source = (sj * nx + si) as usize;
                    if self.nodes[source] == Node::Fluid {
                        self.f[9 * k + q] = self.f_post[9 * source + q];
                    } else {
                        let back = self.f_post[9 * k + OPPOSITE[q]];
                        self.f[9 * k + q] = back;
                        if self.nodes[source] == Node::Cylinder {
                            let c = C[OPPOSITE[q]];
                            force.0 += 2.0 * back * c.0 as f64;
                            force.1 += 2.0 * back * c.1 as f64;
                        }
                    }
                }
            }
        }
        self.force = force;
    }

    /// Zou-He boundary conditions on the first and last column: the three populations that the
    /// wrap-around got from the other end are replaced. Inlet: velocity given. Outlet: density 1.
    fn apply_open_ends(&mut self) {
        let inlet = self.inlet.as_ref().unwrap();
        for j in 0..self.ny {
            let k = j * self.nx;
            if self.nodes[k] == Node::Fluid {
                let ux = inlet[j];
                let f = &mut self.f[9 * k..9 * k + 9];
                f[1] = f[3] + 2.0 / 3.0 * ux;
                f[5] = f[7] - (f[2] - f[4]) / 2.0 + ux / 6.0;
                f[8] = f[6] + (f[2] - f[4]) / 2.0 + ux / 6.0;
            }
            let k = j * self.nx + self.nx - 1;
            if self.nodes[k] == Node::Fluid {
                let f = &mut self.f[9 * k..9 * k + 9];
                let ux = f[0] + f[2] + f[4] + 2.0 * (f[1] + f[5] + f[8]) - 1.0;
                f[3] = f[1] - 2.0 / 3.0 * ux;
                f[7] = f[5] + (f[2] - f[4]) / 2.0 - ux / 6.0;
                f[6] = f[8] - (f[2] - f[4]) / 2.0 - ux / 6.0;
            }
        }
    }

    #[cfg(test)]
    fn density(&self, i: usize, j: usize) -> f64 {
        self.f[9 * (j * self.nx + i)..9 * (j * self.nx + i) + 9].iter().sum()
    }

    /// Velocity (the momentum, since the reference density is 1); zero on solid nodes.
    fn velocity(&self, i: usize, j: usize) -> (f64, f64) {
        let k = j * self.nx + i;
        if self.nodes[k] != Node::Fluid {
            return (0.0, 0.0);
        }
        let f = &self.f[9 * k..9 * k + 9];
        ((0..9).map(|q| f[q] * C[q].0 as f64).sum(), (0..9).map(|q| f[q] * C[q].1 as f64).sum())
    }

    /// Total mass on fluid nodes (tests only).
    #[cfg(test)]
    fn mass(&self) -> f64 {
        let mut sum = 0.0;
        for j in 0..self.ny {
            for i in 0..self.nx {
                if self.nodes[j * self.nx + i] == Node::Fluid {
                    sum += self.density(i, j);
                }
            }
        }
        sum
    }
}

/// The Schaefer-Turek channel (benchmark DFG 2D-1 and 2D-2) for the Reynolds number `reynolds`:
/// 22 D long, 4.1 D high, the cylinder 2 D from the inlet and 2 D above the bottom wall (so
/// slightly off-centre, which starts the shedding), a parabolic inlet and constant pressure at
/// the outlet. Rows 0 and ny - 1 are the walls; fluid row j sits at height y = j - 0.5.
fn channel(reynolds: f64) -> Lattice {
    let n = CELLS_PER_DIAMETER;
    let (nx, ny) = (22 * n, (4.1 * n as f64).round() as usize + 2);
    let (cx, cy, radius) = (2.0 * n as f64 - 0.5, 2.0 * n as f64 + 0.5, n as f64 / 2.0);
    let mut nodes = vec![Node::Fluid; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            if j == 0 || j == ny - 1 {
                nodes[j * nx + i] = Node::Wall;
            } else if (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) < radius * radius {
                nodes[j * nx + i] = Node::Cylinder;
            }
        }
    }
    let height = (ny - 2) as f64;
    let inlet = (0..ny)
        .map(|j| {
            let y = j as f64 - 0.5;
            if j == 0 || j == ny - 1 { 0.0 } else { 4.0 * PEAK_INLET_SPEED * y * (height - y) / (height * height) }
        })
        .collect();
    let nu = MEAN_INLET_SPEED * n as f64 / reynolds;
    Lattice::new(nx, ny, nu, nodes, Some(inlet))
}

/// Drag and lift coefficients, C = 2 F / (mean speed^2 D), from the force of the last step.
fn drag_lift(lattice: &Lattice) -> (f64, f64) {
    let scale = 2.0 / (MEAN_INLET_SPEED * MEAN_INLET_SPEED * CELLS_PER_DIAMETER as f64);
    (lattice.force.0 * scale, lattice.force.1 * scale)
}

/// Length of the recirculation area behind the cylinder in diameters, measured from its rear:
/// where the mean u_x on the two rows just above the centreline turns from negative to positive.
fn recirculation_length(lattice: &Lattice) -> Option<f64> {
    let n = CELLS_PER_DIAMETER;
    let mean_ux = |i: usize| (lattice.velocity(i, 2 * n).0 + lattice.velocity(i, 2 * n + 1).0) / 2.0;
    let start = (2.5 * n as f64) as usize;
    (start..lattice.nx - 1).find(|&i| mean_ux(i) < 0.0 && mean_ux(i + 1) >= 0.0).map(|i| {
        let (a, b) = (mean_ux(i), mean_ux(i + 1));
        (i as f64 + 0.5 - a / (b - a) - 2.5 * n as f64) / n as f64
    })
}

/// Water flows left to right through a narrow channel past a round rod. At Reynolds number 100
/// the wake is unstable: vortices peel off alternately from the top and bottom and drift
/// downstream as a Karman vortex street. The geometry is the Schaefer-Turek benchmark (DFG
/// 2D-2), so the shedding frequency, drag and lift can be compared with published values. R
/// switches to Reynolds 20 (2D-1), where the wake is a steady pair of eddies. Lattice Boltzmann
/// (D2Q9, two-relaxation-time collision), 440 x 84 nodes.
/// Not modelled: resolution (the boundary layer on the rod is 2 cells thick), compressibility
/// (Mach 0.17 to 0.26), the third dimension and side walls, the benchmark's short inlet (it
/// reads St 0.30; the laboratory value with a longer inlet is 0.287), an open stream (the walls
/// are close), reflections from the outlet, and everything else (gravity, free surface,
/// temperature, a rod that moves).
pub struct VortexStreet {
    lattice: Lattice,
    reynolds: f64,
    /// Seconds per lattice step.
    dt: f64,
    t: f64,
    speed: f64,
    steps_this_frame: u32,
    /// Some steps were dropped this frame to keep the frame rate.
    capped: bool,
    previous_lift: f64,
    last_upward_crossing: Option<f64>,
    shedding_period: Option<f64>,
    show_speed: bool,
    texture: OnceCell<Texture2D>,
}

impl VortexStreet {
    pub fn new() -> Self {
        Self::at(REYNOLDS_UNSTEADY)
    }

    fn at(reynolds: f64) -> Self {
        let dx = DIAMETER / CELLS_PER_DIAMETER as f64;
        let mean_speed = reynolds * VISCOSITY / DIAMETER;
        Self {
            lattice: channel(reynolds),
            reynolds,
            dt: MEAN_INLET_SPEED * dx / mean_speed,
            t: 0.0,
            speed: 1.0,
            steps_this_frame: 0,
            capped: false,
            previous_lift: 0.0,
            last_upward_crossing: None,
            shedding_period: None,
            show_speed: false,
            texture: OnceCell::new(),
        }
    }

    /// Mean flow speed, m/s.
    fn mean_speed(&self) -> f64 {
        self.reynolds * VISCOSITY / DIAMETER
    }
}

impl Simulation for VortexStreet {
    fn step(&mut self, _dt: f64) {
        // main.rs asks for more steps when a frame runs long; without a cap a slow machine would
        // fall under 10 fps. With it, simulated time just runs slower than requested.
        if self.steps_this_frame >= MAX_STEPS_PER_FRAME {
            self.capped = true;
            return;
        }
        self.steps_this_frame += 1;
        self.lattice.step();
        self.t += self.dt;

        // Shedding period: the gap between upward zero crossings of the lift.
        let (_, lift) = drag_lift(&self.lattice);
        if self.previous_lift < 0.0 && lift >= 0.0 {
            let crossing = self.t - self.dt * lift / (lift - self.previous_lift);
            if let Some(last) = self.last_upward_crossing {
                self.shedding_period = Some(crossing - last);
            }
            self.last_upward_crossing = Some(crossing);
        }
        self.previous_lift = lift;
    }

    fn is_over(&self) -> bool {
        false
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn world_bounds(&self) -> Rect {
        // 3:2, with the channel below the HUD lines.
        Rect::new(-0.005, -0.045, 0.23, 0.1533)
    }

    fn handle_input(&mut self) {
        // Runs once per frame, before the steps.
        self.steps_this_frame = 0;
        self.capped = false;
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
        if is_key_pressed(KeyCode::V) {
            self.show_speed = !self.show_speed;
        }
        if is_key_pressed(KeyCode::R) {
            let (speed, show_speed) = (self.speed, self.show_speed);
            *self = Self::at(if self.reynolds == REYNOLDS_UNSTEADY { REYNOLDS_STEADY } else { REYNOLDS_UNSTEADY });
            self.speed = speed;
            self.show_speed = show_speed;
        }
    }

    fn draw(&self) {
        let lattice = &self.lattice;
        let (nx, ny) = (lattice.nx, lattice.ny);
        let n = CELLS_PER_DIAMETER as f64;
        let velocities: Vec<(f64, f64)> = (0..nx * ny).map(|k| lattice.velocity(k % nx, k / nx)).collect();
        let at = |i: usize, j: usize| velocities[j * nx + i];

        // One pixel per node, row 0 first. Vorticity: du_y/dx - du_x/dy by centred differences.
        let mut pixels = Vec::with_capacity(4 * nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                if lattice.nodes[j * nx + i] != Node::Fluid {
                    pixels.extend_from_slice(&[64, 64, 64, 255]);
                    continue;
                }
                if self.show_speed {
                    let (ux, uy) = at(i, j);
                    let grey = ((ux.hypot(uy) / (1.5 * PEAK_INLET_SPEED)).clamp(0.0, 1.0) * 255.0) as u8;
                    pixels.extend_from_slice(&[grey, grey, grey, 255]);
                } else {
                    let (left, right) = (at(i.saturating_sub(1), j), at((i + 1).min(nx - 1), j));
                    let (below, above) = (at(i, j.saturating_sub(1)), at(i, (j + 1).min(ny - 1)));
                    let vorticity = (right.1 - left.1) / 2.0 - (above.0 - below.0) / 2.0;
                    let s = (vorticity * n / (6.0 * MEAN_INLET_SPEED)).clamp(-1.0, 1.0);
                    let brightness = s.abs().sqrt();
                    // Orange for anticlockwise, blue for clockwise.
                    let (r, g, b) = if s > 0.0 { (1.0, 0.55, 0.1) } else { (0.15, 0.45, 1.0) };
                    pixels.extend_from_slice(&[(r * brightness * 255.0) as u8, (g * brightness * 255.0) as u8, (b * brightness * 255.0) as u8, 255]);
                }
            }
        }
        let texture = self.texture.get_or_init(|| {
            let texture = Texture2D::from_rgba8(nx as u16, ny as u16, &pixels);
            texture.set_filter(FilterMode::Linear);
            texture
        });
        texture.update_from_bytes(nx as u32, ny as u32, &pixels);
        let dx = (DIAMETER / n) as f32;
        // Node (i, j) is centred at ((i + 0.5) dx, (j - 0.5) dx): row 0 is the bottom wall.
        draw_texture_ex(texture, 0.0, -dx, WHITE, DrawTextureParams { dest_size: Some(vec2(CHANNEL_LENGTH as f32, ny as f32 * dx)), ..Default::default() });
        // A round rod on top of the staircase.
        draw_poly(0.02, 0.02, 48, (DIAMETER / 2.0) as f32, 0.0, GRAY);
    }

    fn hud(&self) -> Vec<String> {
        let (drag, lift) = drag_lift(&self.lattice);
        let third = if self.reynolds == REYNOLDS_UNSTEADY {
            match self.shedding_period {
                Some(period) => {
                    let strouhal = DIAMETER / (period * self.mean_speed());
                    format!("shedding {:.3} Hz   St = {:.3} (benchmark 0.295-0.305)", 1.0 / period, strouhal)
                }
                None => "shedding: waiting for two vortices...".to_string(),
            }
        } else {
            match recirculation_length(&self.lattice) {
                Some(length) => format!("wake length {:.2} D (benchmark 0.842-0.852)", length),
                None => "wake length: not formed yet".to_string(),
            }
        };
        vec![
            format!("t = {:.1} s   speed {}x (UP/DOWN){}", self.t, self.speed, if self.capped { " - capped" } else { "" }),
            format!("Re = {} (R)   1 cm rod in water, mean flow {:.2} cm/s", self.reynolds, 100.0 * self.mean_speed()),
            third,
            format!("C_D = {:.2}   C_L = {:+.2}", drag, lift),
            format!("{} (V)", if self.show_speed { "speed" } else { "vorticity" }),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box periodic in x, with solid rows at 0 and ny - 1.
    fn closed_box(nx: usize, ny: usize, nu: f64, solid_block: Option<(usize, usize, usize, usize)>) -> Lattice {
        let mut nodes = vec![Node::Fluid; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let in_block = solid_block.is_some_and(|(i0, i1, j0, j1)| (i0..=i1).contains(&i) && (j0..=j1).contains(&j));
                if j == 0 || j == ny - 1 || in_block {
                    nodes[j * nx + i] = Node::Wall;
                }
            }
        }
        Lattice::new(nx, ny, nu, nodes, None)
    }

    #[test]
    fn equilibrium_has_the_right_moments() {
        let (rho, ux, uy) = (1.02, 0.05, -0.03);
        let f = equilibrium(rho, ux, uy);
        let sum = |g: &dyn Fn(usize) -> f64| (0..9).map(|q| f[q] * g(q)).sum::<f64>();
        assert!((sum(&|_| 1.0) - rho).abs() < 1e-14);
        assert!((sum(&|q| C[q].0 as f64) - ux).abs() < 1e-14);
        assert!((sum(&|q| C[q].1 as f64) - uy).abs() < 1e-14);
        assert!((sum(&|q| (C[q].0 * C[q].1) as f64) - ux * uy).abs() < 1e-14);
        assert!((sum(&|q| (C[q].0 * C[q].0) as f64) - (rho / 3.0 + ux * ux)).abs() < 1e-14);
    }

    #[test]
    fn mass_is_conserved_in_a_closed_box() {
        let mut lattice = closed_box(64, 34, 0.02, Some((29, 34, 14, 19)));
        for j in 0..34 {
            for i in 0..64 {
                if lattice.nodes[j * 64 + i] == Node::Fluid {
                    lattice.f[9 * (j * 64 + i)..9 * (j * 64 + i) + 9].copy_from_slice(&equilibrium(1.0, 0.05, 0.0));
                }
            }
        }
        let mass0 = lattice.mass();
        for _ in 0..2000 {
            lattice.step();
        }
        assert!(((lattice.mass() - mass0) / mass0).abs() < 1e-11, "mass changed by {}", (lattice.mass() - mass0) / mass0);
    }

    #[test]
    fn shear_wave_decays_at_the_set_viscosity() {
        let (n, nu) = (32, 0.02);
        let mut lattice = Lattice::new(n, n, nu, vec![Node::Fluid; n * n], None);
        let k = std::f64::consts::TAU / n as f64;
        for j in 0..n {
            for i in 0..n {
                let ux = 0.01 * (k * j as f64).sin();
                lattice.f[9 * (j * n + i)..9 * (j * n + i) + 9].copy_from_slice(&equilibrium(1.0, ux, 0.0));
            }
        }
        for _ in 0..1000 {
            lattice.step();
        }
        let amplitude = (0..n).map(|j| lattice.velocity(0, j).0 * (k * j as f64).sin()).sum::<f64>() * 2.0 / n as f64;
        let fitted = -(amplitude / 0.01).ln() / (k * k * 1000.0);
        assert!((fitted / nu - 1.0).abs() < 0.005, "fitted viscosity {fitted}, set {nu}");
    }

    #[test]
    fn channel_flow_is_a_parabola() {
        // Poiseuille flow between walls, driven by a body force added by the test.
        let (height, nu, speed) = (32.0, 0.1, 0.05);
        let mut lattice = closed_box(4, 34, nu, None);
        for k in 0..lattice.nodes.len() {
            if lattice.nodes[k] == Node::Fluid {
                lattice.f[9 * k..9 * k + 9].copy_from_slice(&equilibrium(1.0, 0.0, 0.0));
            }
        }
        let g = 8.0 * nu * speed / (height * height);
        for _ in 0..12_000 {
            for k in 0..lattice.nodes.len() {
                if lattice.nodes[k] == Node::Fluid {
                    for q in 0..9 {
                        lattice.f[9 * k + q] += 3.0 * W[q] * C[q].0 as f64 * g;
                    }
                }
            }
            lattice.step();
        }
        for j in 1..33 {
            let y = j as f64 - 0.5;
            let exact = 4.0 * speed * y * (height - y) / (height * height);
            // The velocity is measured at the middle of the step's kick: add g / 2.
            let measured = lattice.velocity(1, j).0 + g / 2.0;
            assert!((measured - exact).abs() < 5e-4 * speed, "row {j}: {measured} against {exact}");
        }
    }

    #[test]
    fn steady_wake_at_reynolds_20() {
        let mut lattice = channel(20.0);
        for _ in 0..20_000 {
            lattice.step();
        }
        let (drag, lift) = drag_lift(&lattice);
        let length = recirculation_length(&lattice).unwrap();
        assert!((drag / 5.58 - 1.0).abs() < 0.02, "C_D = {drag}");
        assert!((lift - 0.0107).abs() < 0.0012, "C_L = {lift}");
        assert!((length / 0.847 - 1.0).abs() < 0.02, "L_a / D = {length}");
    }

    #[test]
    fn sheds_vortices_at_reynolds_100() {
        let mut lattice = channel(100.0);
        let (mut drags, mut lifts, mut crossings) = (Vec::new(), Vec::new(), Vec::new());
        for step in 0..20_000 {
            lattice.step();
            let (drag, lift) = drag_lift(&lattice);
            if let Some(&previous) = lifts.last() {
                if previous < 0.0 && lift >= 0.0 {
                    crossings.push(step as f64 - lift / (lift - previous));
                }
            }
            drags.push(drag);
            lifts.push(lift);
        }
        // The last five periods.
        let (first, last) = (crossings[crossings.len() - 6], crossings[crossings.len() - 1]);
        let period = (last - first) / 5.0;
        let strouhal = CELLS_PER_DIAMETER as f64 / (period * MEAN_INLET_SPEED);
        let range = first.ceil() as usize..last.floor() as usize;
        let drag_max = drags[range.clone()].iter().cloned().fold(f64::MIN, f64::max);
        let lift_max = lifts[range].iter().cloned().fold(f64::MIN, f64::max);
        assert!((strouhal / 0.300 - 1.0).abs() < 0.02, "St = {strouhal}");
        assert!((drag_max / 3.23 - 1.0).abs() < 0.06, "C_D max = {drag_max}");
        assert!((lift_max / 1.00 - 1.0).abs() < 0.13, "C_L max = {lift_max}");
    }
}
