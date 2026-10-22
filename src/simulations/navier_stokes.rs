use std::cell::OnceCell;
use std::f64::consts::{PI, TAU};

use macroquad::prelude::*;
use macroquad::rand::RandGenerator;

use crate::simulation::Simulation;

// Units: dimensionless. Each scene measures lengths and speeds in its own scale (the box and the
// lid speed, the shear layer's half-thickness and half-speed difference, the layer depth and the
// free-fall speed), and time in length / speed. All that is left of the fluid is the Reynolds
// number, and for convection the Rayleigh and Prandtl numbers.

/// Cells across the cavity at Re 1000 and Re 100.
const CAVITY_CELLS_FINE: usize = 128;
const CAVITY_CELLS_COARSE: usize = 64;
const REYNOLDS_CAVITY: [f64; 2] = [1000.0, 100.0];

/// The shear layer u = tanh(y): its most unstable wave number, and growth rate without viscosity
/// (Michalke 1964). The box holds two of those waves.
const SHEAR_WAVE_NUMBER: f64 = 0.4446;
const SHEAR_GROWTH_INVISCID: f64 = 0.1897;
const SHEAR_REYNOLDS: f64 = 1000.0;
/// Starting amplitudes of v for the fastest wave and for its subharmonic (which makes pairs merge).
const SHEAR_KICK: f64 = 1e-4;
const SHEAR_KICK_SUBHARMONIC: f64 = 2e-5;
/// Above this amplitude of v the wave is no longer small, and linear theory stops applying.
const SHEAR_LINEAR_LIMIT: f64 = 0.03;

/// Water (Pr 7) heated from below. Rayleigh-Benard onset between rigid walls: Ra 1707.76 at wave
/// number 3.117 (Pellew and Southwell 1940).
const CONVECTION_RAYLEIGH: f64 = 1e5;
const CONVECTION_PRANDTL: f64 = 7.0;
const CONVECTION_WIDTH: f64 = 4.0;
const CONVECTION_CELLS_HIGH: usize = 64;

const MAX_STEPS_PER_FRAME: u32 = 12;
const MIN_SPEED: f64 = 0.25;
const MAX_SPEED: f64 = 8.0;

/// Ghia, Ghia and Shin (1982), tables I and II: u along the vertical centre-line (y, Re 100,
/// Re 1000) and v along the horizontal one (x, Re 100, Re 1000), from a 129 x 129 grid.
const GHIA_U: [(f64, f64, f64); 17] = [
    (1.0000, 1.00000, 1.00000),
    (0.9766, 0.84123, 0.65928),
    (0.9688, 0.78871, 0.57492),
    (0.9609, 0.73722, 0.51117),
    (0.9531, 0.68717, 0.46604),
    (0.8516, 0.23151, 0.33304),
    (0.7344, 0.00332, 0.18719),
    (0.6172, -0.13641, 0.05702),
    (0.5000, -0.20581, -0.06080),
    (0.4531, -0.21090, -0.10648),
    (0.2813, -0.15662, -0.27805),
    (0.1719, -0.10150, -0.38289),
    (0.1016, -0.06434, -0.29730),
    (0.0703, -0.04775, -0.22220),
    (0.0625, -0.04192, -0.20196),
    (0.0547, -0.03717, -0.18109),
    (0.0000, 0.00000, 0.00000),
];
const GHIA_V: [(f64, f64, f64); 17] = [
    (1.0000, 0.00000, 0.00000),
    (0.9688, -0.05906, -0.21388),
    (0.9609, -0.07391, -0.27669),
    (0.9531, -0.08864, -0.33714),
    (0.9453, -0.10313, -0.39188),
    (0.9063, -0.16914, -0.51550),
    (0.8594, -0.22445, -0.42665),
    (0.8047, -0.24533, -0.31966),
    (0.5000, 0.05454, 0.02526),
    (0.2344, 0.17527, 0.32235),
    (0.2266, 0.17507, 0.33075),
    (0.1563, 0.16077, 0.37095),
    (0.0938, 0.12317, 0.32627),
    (0.0781, 0.10890, 0.30353),
    (0.0703, 0.10091, 0.29012),
    (0.0625, 0.09233, 0.27485),
    (0.0000, 0.00000, 0.00000),
];
/// Ghia's primary vortex: stream function and centre (x, y), for Re 100 and Re 1000.
const GHIA_VORTEX: [(f64, f64, f64); 2] = [(-0.103423, 0.6172, 0.7344), (-0.117929, 0.5313, 0.5625)];

/// In-place iterative radix-2 FFT (Cooley-Tukey) of one power-of-two length, as separate real
/// and imaginary parts. Forward sign e^(-2 pi i jk/n); unnormalised both ways.
struct Fft {
    n: usize,
    /// e^(-2 pi i k / n) for k < n / 2.
    twiddles: Vec<(f64, f64)>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT length {n} is not a power of two");
        let twiddles = (0..n / 2).map(|k| (-TAU * k as f64 / n as f64).sin_cos()).map(|(s, c)| (c, s)).collect();
        Self { n, twiddles }
    }

    fn transform(&self, re: &mut [f64], im: &mut [f64], inverse: bool) {
        let n = self.n;
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.twiddles[k * stride];
                    let wi = if inverse { -wi } else { wi };
                    let (a, b) = (start + k, start + k + len / 2);
                    let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
            len <<= 1;
        }
    }
}

/// Solves the pressure equation L phi = rhs, with L the 5-point Laplacian on the cell centres,
/// zero normal gradient at walls, and periodic in x if the flow is. A direct method: the 1D
/// Laplacian in x has known orthonormal eigenvectors, so writing each row in them leaves one
/// tridiagonal system in y per mode. Between walls they are the cosines
///   c_k cos(pi k (i + 1/2) / n),   k = 0..n,
/// and if periodic the constant, then cos and sin of 2 pi w i / n for w = 1, 2, ... in turn, and
/// cos(pi i) last; c is sqrt(1/n) for the constant and cos(pi i), sqrt(2/n) otherwise. Both
/// transforms go through an FFT (of the row mirrored to twice its length between walls).
struct Poisson {
    nx: usize,
    ny: usize,
    periodic: bool,
    fft: Fft,
    /// Eigenvalues times h^2 (0 for the constant, negative otherwise).
    eigen: Vec<f64>,
}

impl Poisson {
    fn new(nx: usize, ny: usize, periodic: bool) -> Self {
        let n = nx as f64;
        let eigen = (0..nx)
            .map(|k| {
                if periodic {
                    -4.0 * (PI * k.div_ceil(2) as f64 / n).sin().powi(2)
                } else {
                    -4.0 * (PI * k as f64 / (2.0 * n)).sin().powi(2)
                }
            })
            .collect();
        let fft = Fft::new(if periodic { nx } else { 2 * nx });
        Self { nx, ny, periodic, fft, eigen }
    }

    /// Normalisation of eigenvector k.
    fn norm(&self, k: usize) -> f64 {
        let n = self.nx as f64;
        let alone = k == 0 || (self.periodic && k == self.nx - 1);
        if alone { (1.0 / n).sqrt() } else { (2.0 / n).sqrt() }
    }

    /// Coefficients of `row` in the eigenvectors.
    fn forward(&self, row: &[f64], out: &mut [f64]) {
        let n = self.nx;
        let (mut re, mut im) = (vec![0.0; self.fft.n], vec![0.0; self.fft.n]);
        re[..n].copy_from_slice(row);
        if !self.periodic {
            // Mirrored: Y_k = 2 e^(i pi k / 2n) sum_i x_i cos(pi k (i + 1/2) / n).
            for i in 0..n {
                re[2 * n - 1 - i] = row[i];
            }
        }
        self.fft.transform(&mut re, &mut im, false);
        for k in 0..n {
            out[k] = self.norm(k)
                * if self.periodic {
                    let w = k.div_ceil(2);
                    if k == 0 || k % 2 == 1 { re[w] } else { -im[w] }
                } else {
                    let (s, c) = (PI * k as f64 / (2.0 * n as f64)).sin_cos();
                    (c * re[k] + s * im[k]) / 2.0
                };
        }
    }

    /// The row with coefficients `coefficients`: x_i = Re(sum_k G_k e^(2 pi i k i / N)).
    fn inverse(&self, coefficients: &[f64], out: &mut [f64]) {
        let n = self.nx;
        let (mut re, mut im) = (vec![0.0; self.fft.n], vec![0.0; self.fft.n]);
        for k in 0..n {
            let a = self.norm(k) * coefficients[k];
            if self.periodic {
                // a cos + b sin = Re((a - i b) e^(i angle)).
                let w = k.div_ceil(2);
                if k == 0 || k % 2 == 1 {
                    re[w] += a;
                } else {
                    im[w] -= a;
                }
            } else {
                let (s, c) = (PI * k as f64 / (2.0 * n as f64)).sin_cos();
                re[k] = a * c;
                im[k] = a * s;
            }
        }
        self.fft.transform(&mut re, &mut im, true);
        out.copy_from_slice(&re[..n]);
    }

    /// `rhs` is already multiplied by h^2. The constant mode is pinned to zero at the bottom row,
    /// which fixes the free constant in phi (rhs must sum to zero, as a divergence does).
    fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let mut hat = vec![0.0; nx * ny];
        for j in 0..ny {
            self.forward(&rhs[j * nx..(j + 1) * nx], &mut hat[j * nx..(j + 1) * nx]);
        }
        // Thomas algorithm for every mode at once: phi[j-1] + (lambda - 2) phi[j] + phi[j+1] = rhs,
        // where a missing neighbour at a wall is replaced by phi[j] itself.
        let diagonal = |k: usize, j: usize| {
            if self.eigen[k] == 0.0 && j == 0 {
                return 1.0;
            }
            let walls = (j == 0) as usize + (j == ny - 1) as usize;
            self.eigen[k] - 2.0 + walls as f64
        };
        let mut c_prime = vec![0.0; nx * ny];
        for k in 0..nx {
            if self.eigen[k] == 0.0 {
                hat[k] = 0.0;
                c_prime[k] = 0.0;
            } else {
                let b = diagonal(k, 0);
                c_prime[k] = 1.0 / b;
                hat[k] /= b;
            }
        }
        for j in 1..ny {
            for k in 0..nx {
                let m = diagonal(k, j) - c_prime[(j - 1) * nx + k];
                c_prime[j * nx + k] = 1.0 / m;
                hat[j * nx + k] = (hat[j * nx + k] - hat[(j - 1) * nx + k]) / m;
            }
        }
        for j in (0..ny - 1).rev() {
            for k in 0..nx {
                hat[j * nx + k] -= c_prime[j * nx + k] * hat[(j + 1) * nx + k];
            }
        }
        let mut phi = vec![0.0; nx * ny];
        for j in 0..ny {
            self.inverse(&hat[j * nx..(j + 1) * nx], &mut phi[j * nx..(j + 1) * nx]);
        }
        phi
    }
}

/// The velocity (and, for convection, temperature) on a staggered grid of nx x ny square cells
/// of side h (a MAC grid): u on the left face of each cell, v on the bottom face, temperature at
/// the centre. Index u as `j * (nx + 1) + i` (i up to nx) and v as `j * nx + i` (j up to ny).
/// The top and bottom are walls; the sides are walls or periodic (then u's column nx repeats
/// column 0). No macroquad types, so tests can build it without a window.
#[derive(Clone)]
struct Fields {
    u: Vec<f64>,
    v: Vec<f64>,
    temperature: Vec<f64>,
}

struct Flow {
    nx: usize,
    ny: usize,
    h: f64,
    periodic: bool,
    /// Top and bottom walls let the fluid slide (no shear stress) instead of holding it.
    free_slip: bool,
    /// Speed of the top wall along x: the cavity's lid.
    lid: f64,
    /// Kinematic viscosity (1 / Re).
    nu: f64,
    /// Thermal diffusivity, the temperatures of the bottom and top walls, and the buoyancy:
    /// dv/dt gets `buoyancy * temperature`. Zero buoyancy and no temperature without convection.
    kappa: f64,
    t_bottom: f64,
    t_top: f64,
    buoyancy: f64,
    fields: Fields,
    poisson: Poisson,
    dt: f64,
    t: f64,
}

impl Flow {
    fn new(nx: usize, ny: usize, h: f64, periodic: bool, free_slip: bool, nu: f64, dt: f64) -> Self {
        Self {
            nx,
            ny,
            h,
            periodic,
            free_slip,
            lid: 0.0,
            nu,
            kappa: 0.0,
            t_bottom: 0.0,
            t_top: 0.0,
            buoyancy: 0.0,
            fields: Fields { u: vec![0.0; (nx + 1) * ny], v: vec![0.0; nx * (ny + 1)], temperature: Vec::new() },
            poisson: Poisson::new(nx, ny, periodic),
            dt,
            t: 0.0,
        }
    }

    fn has_temperature(&self) -> bool {
        !self.fields.temperature.is_empty()
    }

    /// u at face (i, j), with ghost values beyond the walls: across the top and bottom walls the
    /// value that makes the average on the wall equal the wall's speed (no slip), or the same value
    /// (free slip).
    fn u_at(&self, f: &Fields, i: isize, j: isize) -> f64 {
        let (nx, ny) = (self.nx as isize, self.ny as isize);
        let i = if self.periodic { i.rem_euclid(nx) } else { i.clamp(0, nx) };
        if j < 0 {
            let inside = f.u[i as usize];
            return if self.free_slip { inside } else { -inside };
        }
        if j >= ny {
            let inside = f.u[((ny - 1) * (nx + 1) + i) as usize];
            return if self.free_slip { inside } else { 2.0 * self.lid - inside };
        }
        f.u[(j * (nx + 1) + i) as usize]
    }

    /// v at face (i, j); beyond side walls the ghost value makes v zero on the wall (no slip).
    fn v_at(&self, f: &Fields, i: isize, j: isize) -> f64 {
        let nx = self.nx as isize;
        let j = j.clamp(0, self.ny as isize);
        if self.periodic {
            return f.v[(j * nx + i.rem_euclid(nx)) as usize];
        }
        if i < 0 {
            -f.v[(j * nx) as usize]
        } else if i >= nx {
            -f.v[(j * nx + nx - 1) as usize]
        } else {
            f.v[(j * nx + i) as usize]
        }
    }

    /// Temperature at cell (i, j); beyond the top and bottom walls the ghost value makes the
    /// average on the wall equal the wall's temperature; side walls (if any) are insulating.
    fn temperature_at(&self, f: &Fields, i: isize, j: isize) -> f64 {
        let (nx, ny) = (self.nx as isize, self.ny as isize);
        let i = if self.periodic { i.rem_euclid(nx) } else { i.clamp(0, nx - 1) };
        if j < 0 {
            return 2.0 * self.t_bottom - f.temperature[i as usize];
        }
        if j >= ny {
            return 2.0 * self.t_top - f.temperature[((ny - 1) * nx + i) as usize];
        }
        f.temperature[(j * nx + i) as usize]
    }

    /// Range of u faces that are unknowns: all of them if periodic (column nx is a copy), else
    /// the ones between the side walls.
    fn u_columns(&self) -> std::ops::Range<usize> {
        if self.periodic { 0..self.nx } else { 1..self.nx }
    }

    /// Time derivatives without the pressure: -(u . grad) u + nu laplacian u + buoyancy, and for
    /// the temperature -(u . grad) T + kappa laplacian T. The advection is in conservative form
    /// with centred averages (Harlow and Welch's MAC scheme), which neither adds nor removes
    /// kinetic energy, so all the damping is the physical viscosity.
    fn tendency(&self, f: &Fields) -> Fields {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let mut out = Fields { u: vec![0.0; f.u.len()], v: vec![0.0; f.v.len()], temperature: vec![0.0; f.temperature.len()] };
        let u = |i: usize, j: usize, di: isize, dj: isize| self.u_at(f, i as isize + di, j as isize + dj);
        let v = |i: usize, j: usize, di: isize, dj: isize| self.v_at(f, i as isize + di, j as isize + dj);

        for j in 0..ny {
            for i in self.u_columns() {
                let here = u(i, j, 0, 0);
                let east = (here + u(i, j, 1, 0)) / 2.0;
                let west = (u(i, j, -1, 0) + here) / 2.0;
                let (u_north, v_north) = ((here + u(i, j, 0, 1)) / 2.0, (v(i, j, -1, 1) + v(i, j, 0, 1)) / 2.0);
                let (u_south, v_south) = ((u(i, j, 0, -1) + here) / 2.0, (v(i, j, -1, 0) + v(i, j, 0, 0)) / 2.0);
                let advection = (east * east - west * west + u_north * v_north - u_south * v_south) / h;
                let laplacian = (u(i, j, 1, 0) + u(i, j, -1, 0) + u(i, j, 0, 1) + u(i, j, 0, -1) - 4.0 * here) / (h * h);
                out.u[j * (nx + 1) + i] = -advection + self.nu * laplacian;
            }
            if self.periodic {
                out.u[j * (nx + 1) + nx] = out.u[j * (nx + 1)];
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let here = v(i, j, 0, 0);
                let north = (here + v(i, j, 0, 1)) / 2.0;
                let south = (v(i, j, 0, -1) + here) / 2.0;
                let (u_east, v_east) = ((u(i, j, 1, -1) + u(i, j, 1, 0)) / 2.0, (here + v(i, j, 1, 0)) / 2.0);
                let (u_west, v_west) = ((u(i, j, 0, -1) + u(i, j, 0, 0)) / 2.0, (v(i, j, -1, 0) + here) / 2.0);
                let advection = (north * north - south * south + u_east * v_east - u_west * v_west) / h;
                let laplacian = (v(i, j, 1, 0) + v(i, j, -1, 0) + v(i, j, 0, 1) + v(i, j, 0, -1) - 4.0 * here) / (h * h);
                let mut dv = -advection + self.nu * laplacian;
                if self.has_temperature() {
                    dv += self.buoyancy * (f.temperature[(j - 1) * nx + i] + f.temperature[j * nx + i]) / 2.0;
                }
                out.v[j * nx + i] = dv;
            }
        }
        if self.has_temperature() {
            let t = |i: usize, j: usize, di: isize, dj: isize| self.temperature_at(f, i as isize + di, j as isize + dj);
            for j in 0..ny {
                for i in 0..nx {
                    let here = t(i, j, 0, 0);
                    let flux_x = u(i, j, 1, 0) * (here + t(i, j, 1, 0)) / 2.0 - u(i, j, 0, 0) * (t(i, j, -1, 0) + here) / 2.0;
                    let flux_y = v(i, j, 0, 1) * (here + t(i, j, 0, 1)) / 2.0 - v(i, j, 0, 0) * (t(i, j, 0, -1) + here) / 2.0;
                    let laplacian = (t(i, j, 1, 0) + t(i, j, -1, 0) + t(i, j, 0, 1) + t(i, j, 0, -1) - 4.0 * here) / (h * h);
                    out.temperature[j * nx + i] = -(flux_x + flux_y) / h + self.kappa * laplacian;
                }
            }
        }
        out
    }

    /// Divergence of the velocity in each cell, times h (the net outflow per unit depth).
    fn divergence_h(&self, f: &Fields) -> Vec<f64> {
        let nx = self.nx;
        let mut div = vec![0.0; nx * self.ny];
        for j in 0..self.ny {
            for i in 0..nx {
                div[j * nx + i] = f.u[j * (nx + 1) + i + 1] - f.u[j * (nx + 1) + i] + f.v[(j + 1) * nx + i] - f.v[j * nx + i];
            }
        }
        div
    }

    /// Removes the divergent part of the velocity (the pressure's job): solve L phi = div u and
    /// subtract grad phi. What is left is divergence-free to rounding.
    fn project(&self, f: &mut Fields) {
        let nx = self.nx;
        let rhs: Vec<f64> = self.divergence_h(f).iter().map(|d| d * self.h).collect();
        let phi = self.poisson.solve(&rhs);
        let phi_at = |i: isize, j: usize| phi[j * nx + i.rem_euclid(nx as isize) as usize];
        for j in 0..self.ny {
            for i in self.u_columns() {
                f.u[j * (nx + 1) + i] -= (phi_at(i as isize, j) - phi_at(i as isize - 1, j)) / self.h;
            }
            if self.periodic {
                f.u[j * (nx + 1) + nx] = f.u[j * (nx + 1)];
            }
        }
        for j in 1..self.ny {
            for i in 0..nx {
                f.v[j * nx + i] -= (phi[j * nx + i] - phi[(j - 1) * nx + i]) / self.h;
            }
        }
    }

    /// `a f1 + b (f2 + dt d)`, field by field.
    fn combine(a: f64, f1: &Fields, b: f64, f2: &Fields, dt: f64, d: &Fields) -> Fields {
        let mix = |x: &[f64], y: &[f64], dx: &[f64]| x.iter().zip(y).zip(dx).map(|((x, y), dx)| a * x + b * (y + dt * dx)).collect();
        Fields { u: mix(&f1.u, &f2.u, &d.u), v: mix(&f1.v, &f2.v, &d.v), temperature: mix(&f1.temperature, &f2.temperature, &d.temperature) }
    }

    /// One step of the three-stage strong-stability-preserving Runge-Kutta method (Shu and
    /// Osher), third order, projecting after every stage. Its stability region covers part of the
    /// imaginary axis, which the energy-conserving advection needs.
    fn step(&mut self) {
        let dt = self.dt;
        let start = self.fields.clone();
        let mut one = Self::combine(0.0, &start, 1.0, &start, dt, &self.tendency(&start));
        self.project(&mut one);
        let mut two = Self::combine(0.75, &start, 0.25, &one, dt, &self.tendency(&one));
        self.project(&mut two);
        let mut three = Self::combine(1.0 / 3.0, &start, 2.0 / 3.0, &two, dt, &self.tendency(&two));
        self.project(&mut three);
        self.fields = three;
        self.t += dt;
    }

    /// Velocity at a point (bilinear in each staggered component), for the tracers and plots.
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let f = &self.fields;
        let bilinear = |fx: f64, fy: f64, at: &dyn Fn(isize, isize) -> f64| {
            let (i, j) = (fx.floor(), fy.floor());
            let (sx, sy) = (fx - i, fy - j);
            let (i, j) = (i as isize, j as isize);
            (1.0 - sy) * ((1.0 - sx) * at(i, j) + sx * at(i + 1, j)) + sy * ((1.0 - sx) * at(i, j + 1) + sx * at(i + 1, j + 1))
        };
        let u = bilinear(x / self.h, y / self.h - 0.5, &|i, j| self.u_at(f, i, j));
        let v = bilinear(x / self.h - 0.5, y / self.h, &|i, j| self.v_at(f, i, j));
        (u, v)
    }

    /// Vorticity dv/dx - du/dy at the centre of each cell, from the four surrounding corners.
    fn vorticity(&self) -> Vec<f64> {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let f = &self.fields;
        let corner = |i: isize, j: isize| (self.v_at(f, i, j) - self.v_at(f, i - 1, j)) / h - (self.u_at(f, i, j) - self.u_at(f, i, j - 1)) / h;
        let mut out = vec![0.0; nx * ny];
        for j in 0..ny as isize {
            for i in 0..nx as isize {
                out[(j as usize) * nx + i as usize] = (corner(i, j) + corner(i + 1, j) + corner(i, j + 1) + corner(i + 1, j + 1)) / 4.0;
            }
        }
        out
    }

    /// Largest |div u| over the cells (should be rounding).
    fn max_divergence(&self) -> f64 {
        self.divergence_h(&self.fields).iter().map(|d| (d / self.h).abs()).fold(0.0, f64::max)
    }

    fn kinetic_energy(&self) -> f64 {
        let f = &self.fields;
        let nx = self.nx;
        let mut sum = 0.0;
        for j in 0..self.ny {
            for i in 0..nx {
                sum += f.u[j * (nx + 1) + i].powi(2);
            }
        }
        for j in 1..self.ny {
            for i in 0..nx {
                sum += f.v[j * nx + i].powi(2);
            }
        }
        0.5 * sum * self.h * self.h
    }

    /// Stream function at the cell corners, `psi[j * (nx + 1) + i]` at (i h, j h), zero on the
    /// bottom wall: psi(y) is the integral of u from the bottom.
    fn stream_function(&self) -> Vec<f64> {
        let nx = self.nx;
        let mut psi = vec![0.0; (nx + 1) * (self.ny + 1)];
        for j in 0..self.ny {
            for i in 0..=nx {
                psi[(j + 1) * (nx + 1) + i] = psi[j * (nx + 1) + i] + self.fields.u[j * (nx + 1) + i] * self.h;
            }
        }
        psi
    }

    /// Heat carried through the bottom wall relative to conduction alone: the Nusselt number.
    fn nusselt(&self) -> f64 {
        let nx = self.nx;
        let gradient: f64 = (0..nx).map(|i| (self.t_bottom - self.fields.temperature[i]) / (self.h / 2.0)).sum::<f64>() / nx as f64;
        let depth = self.ny as f64 * self.h;
        gradient / ((self.t_bottom - self.t_top) / depth)
    }
}

/// Amplitude of the Fourier mode `mode` along x of v, root-mean-square over the rows.
fn mode_amplitude(flow: &Flow, mode: usize) -> f64 {
    let nx = flow.nx;
    let mut sum = 0.0;
    for j in 1..flow.ny {
        let (mut c, mut s) = (0.0, 0.0);
        for i in 0..nx {
            let angle = TAU * (mode * i) as f64 / nx as f64;
            let v = flow.fields.v[j * nx + i];
            c += v * angle.cos();
            s += v * angle.sin();
        }
        sum += (c * c + s * s) * (2.0 / nx as f64).powi(2);
    }
    (sum / (flow.ny - 1) as f64).sqrt()
}

/// The lid-driven cavity: a square box of water whose lid slides to the right.
fn cavity(reynolds: f64) -> Flow {
    let n = if reynolds > 500.0 { CAVITY_CELLS_FINE } else { CAVITY_CELLS_COARSE };
    let h = 1.0 / n as f64;
    let nu = 1.0 / reynolds;
    // Advection: 0.4 h per unit speed. Diffusion (2D, explicit): 0.2 h^2 / nu.
    let dt = (0.4 * h).min(0.2 * h * h / nu);
    let mut flow = Flow::new(n, n, h, false, false, nu, dt);
    flow.lid = 1.0;
    flow
}

/// A shear layer u = tanh(y), periodic in x, two of its fastest-growing waves long, between
/// free-slip walls at y = +- (box length) / 4, nudged by a small wave and its subharmonic.
/// Coordinates in the flow start at the bottom wall, so y = 0 here is the flow's middle row.
fn shear_layer(nx: usize, reynolds: f64, kick: f64, kick_subharmonic: f64) -> Flow {
    let length = 2.0 * TAU / SHEAR_WAVE_NUMBER;
    let ny = nx / 2;
    let h = length / nx as f64;
    let mut flow = Flow::new(nx, ny, h, true, true, 1.0 / reynolds, 0.4 * h);
    let half = ny as f64 * h / 2.0;
    for j in 0..ny {
        let y = (j as f64 + 0.5) * h - half;
        for i in 0..=nx {
            flow.fields.u[j * (nx + 1) + i] = y.tanh();
        }
    }
    // The perturbation: v ~ sin(k x) exp(-y^2 / 2) (and the subharmonic); the projection makes it
    // divergence-free.
    for j in 1..ny {
        let y = j as f64 * h - half;
        for i in 0..nx {
            let x = (i as f64 + 0.5) * h;
            let envelope = (-y * y / 2.0).exp();
            flow.fields.v[j * nx + i] =
                envelope * (kick * (SHEAR_WAVE_NUMBER * x).sin() + kick_subharmonic * (SHEAR_WAVE_NUMBER / 2.0 * x).sin());
        }
    }
    let mut fields = flow.fields.clone();
    flow.project(&mut fields);
    flow.fields = fields;
    flow
}

/// Rayleigh-Benard convection in free-fall units: lengths in the depth, speeds in
/// sqrt(g alpha dT depth), temperature in dT (bottom +1/2, top -1/2). Then nu = sqrt(Pr / Ra),
/// kappa = 1 / sqrt(Ra Pr) and the buoyancy is the temperature. Starts conducting (linear
/// temperature) with a little noise. `ny` cells make the depth, so the layer is nx / ny wide.
fn convection(nx: usize, ny: usize, rayleigh: f64, prandtl: f64, seed: u64) -> Flow {
    let h = 1.0 / ny as f64;
    let nu = (prandtl / rayleigh).sqrt();
    let kappa = 1.0 / (rayleigh * prandtl).sqrt();
    // Advection: speeds stay below about 0.5. Diffusion: the larger of nu and kappa.
    let dt = (0.5 * h).min(0.25 * h * h / nu.max(kappa));
    let mut flow = Flow::new(nx, ny, h, true, false, nu, dt);
    flow.kappa = kappa;
    flow.buoyancy = 1.0;
    flow.t_bottom = 0.5;
    flow.t_top = -0.5;
    let rng = RandGenerator::new();
    rng.srand(seed);
    flow.fields.temperature = (0..nx * ny)
        .map(|k| {
            let y = ((k / nx) as f64 + 0.5) * h;
            0.5 - y + 1e-3 * rng.gen_range(-1.0, 1.0)
        })
        .collect();
    flow
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Cavity,
    ShearLayer,
    Convection,
}

/// Passive tracers carried by the flow, each with a colour, drawn into a texture.
struct Tracers {
    positions: Vec<(f64, f64)>,
    colors: Vec<[u8; 3]>,
}

impl Tracers {
    /// One tracer per `spacing` in a grid over the box, jittered so the grid doesn't show as a
    /// pattern, coloured by `color(x, y)`.
    fn grid(width: f64, height: f64, spacing: f64, color: impl Fn(f64, f64) -> [u8; 3]) -> Self {
        let (mut positions, mut colors) = (Vec::new(), Vec::new());
        let (cols, rows) = ((width / spacing) as usize, (height / spacing) as usize);
        let rng = RandGenerator::new();
        rng.srand(5);
        for r in 0..rows {
            for c in 0..cols {
                let (x, y) = ((c as f64 + rng.gen_range(0.25, 0.75)) * spacing, (r as f64 + rng.gen_range(0.25, 0.75)) * spacing);
                positions.push((x, y));
                colors.push(color(x, y));
            }
        }
        Self { positions, colors }
    }

    /// Midpoint rule through the velocity field.
    fn advance(&mut self, flow: &Flow, dt: f64) {
        let (width, height) = (flow.nx as f64 * flow.h, flow.ny as f64 * flow.h);
        for p in &mut self.positions {
            let (u, v) = flow.velocity(p.0, p.1);
            let (u, v) = flow.velocity(p.0 + 0.5 * dt * u, p.1 + 0.5 * dt * v);
            p.0 += dt * u;
            p.1 = (p.1 + dt * v).clamp(0.0, height);
            p.0 = if flow.periodic { p.0.rem_euclid(width) } else { p.0.clamp(0.0, width) };
        }
    }
}

/// Viscous flow of an incompressible liquid, from the Navier-Stokes equations
///   du/dt + (u . grad) u = -grad p + nu laplacian u (+ buoyancy),   div u = 0,
/// in three scenes (keys 1-3):
/// 1. A square cavity whose lid slides to the right (Re 1000, R for 100). The flow settles into
///    one big eddy with smaller ones in the corners; the profiles through the centre are drawn
///    against Ghia, Ghia and Shin's (1982) benchmark.
/// 2. Two layers sliding past each other (a tanh shear layer, Re 1000): the Kelvin-Helmholtz
///    instability rolls the interface into vortices, which then pair up. The growth rate of the
///    waves is shown against linear theory.
/// 3. Water heated from below (Rayleigh-Benard convection, Ra 1e5, Pr 7): warm plumes rise, cold
///    ones sink, and the HUD shows how much heat the motion carries (the Nusselt number).
///
/// Finite differences on a staggered (MAC) grid: energy-conserving central advection, explicit
/// viscosity, third-order Runge-Kutta, and a direct pressure solve that keeps the flow exactly
/// divergence-free. Colour is vorticity (orange anticlockwise, blue clockwise) or temperature;
/// the dots in the shear layer are tracers carried by the flow.
/// Not modelled: the third dimension (turbulence in 3D cascades energy the other way), resolution
/// (grids of 64 to 256 cells; Re 1000 is about the limit for 128), sound (the liquid is exactly
/// incompressible), and, for convection, any temperature dependence of the water besides its
/// density (Boussinesq).
pub struct NavierStokes {
    kind: Kind,
    reynolds: f64,
    flow: Flow,
    tracers: Tracers,
    speed: f64,
    steps_this_frame: u32,
    capped: bool,
    /// Shear layer: ln of the wave's amplitude at the last few samples (time, value).
    growth_samples: Vec<(f64, f64)>,
    background: OnceCell<Texture2D>,
    dots: OnceCell<Texture2D>,
}

const ORANGE: [u8; 3] = [255, 150, 40];
const BLUE: [u8; 3] = [60, 140, 255];

impl NavierStokes {
    pub fn new() -> Self {
        Self::scene(Kind::Cavity, REYNOLDS_CAVITY[0])
    }

    fn scene(kind: Kind, reynolds: f64) -> Self {
        let (flow, tracers) = match kind {
            Kind::Cavity => (cavity(reynolds), Tracers { positions: Vec::new(), colors: Vec::new() }),
            Kind::ShearLayer => {
                let flow = shear_layer(256, SHEAR_REYNOLDS, SHEAR_KICK, SHEAR_KICK_SUBHARMONIC);
                let (width, height) = (flow.nx as f64 * flow.h, flow.ny as f64 * flow.h);
                let tracers = Tracers::grid(width, height, flow.h / 1.5, |_, y| if y > height / 2.0 { ORANGE } else { BLUE });
                (flow, tracers)
            }
            Kind::Convection => {
                let nx = (CONVECTION_WIDTH * CONVECTION_CELLS_HIGH as f64) as usize;
                let flow = convection(nx, CONVECTION_CELLS_HIGH, CONVECTION_RAYLEIGH, CONVECTION_PRANDTL, 11);
                (flow, Tracers { positions: Vec::new(), colors: Vec::new() })
            }
        };
        Self {
            kind,
            reynolds,
            flow,
            tracers,
            speed: 1.0,
            steps_this_frame: 0,
            capped: false,
            growth_samples: Vec::new(),
            background: OnceCell::new(),
            dots: OnceCell::new(),
        }
    }

    /// Size of the box in world units, and where its bottom-left corner is drawn.
    fn box_rect(&self) -> Rect {
        let (width, height) = ((self.flow.nx as f64 * self.flow.h) as f32, (self.flow.ny as f64 * self.flow.h) as f32);
        match self.kind {
            Kind::ShearLayer => Rect::new(0.0, -height / 2.0, width, height),
            _ => Rect::new(0.0, 0.0, width, height),
        }
    }

    /// Current growth rate of the shear layer's wave: slope of ln(amplitude) over the last
    /// time unit.
    fn growth_rate(&self) -> Option<f64> {
        let (t1, a1) = *self.growth_samples.last()?;
        let (t0, a0) = *self.growth_samples.iter().find(|(t, _)| t1 - t <= 1.0)?;
        (t1 - t0 > 0.5).then(|| (a1 - a0) / (t1 - t0))
    }

    /// Ghia's columns for this Reynolds number: 1 for Re 100, 2 for Re 1000.
    fn ghia_index(&self) -> usize {
        if self.reynolds > 500.0 { 1 } else { 0 }
    }

    /// Largest difference from Ghia's centre-line velocities, u and v.
    fn ghia_deviation(&self) -> (f64, f64) {
        let column = self.ghia_index();
        let pick = |row: &(f64, f64, f64)| if column == 0 { row.1 } else { row.2 };
        let du = GHIA_U.iter().map(|row| (self.flow.velocity(0.5, row.0).0 - pick(row)).abs()).fold(0.0, f64::max);
        let dv = GHIA_V.iter().map(|row| (self.flow.velocity(row.0, 0.5).1 - pick(row)).abs()).fold(0.0, f64::max);
        (du, dv)
    }

    /// The primary vortex: the stream function's extremum and where it is.
    fn primary_vortex(&self) -> (f64, f64, f64) {
        let psi = self.flow.stream_function();
        let n = self.flow.nx;
        let (k, value) = psi.iter().enumerate().fold((0, 0.0), |best, (k, &p)| if p < best.1 { (k, p) } else { best });
        let (i, j) = (k % (n + 1), k / (n + 1));
        (value, i as f64 * self.flow.h, j as f64 * self.flow.h)
    }

    fn draw_background(&self) {
        let flow = &self.flow;
        let (nx, ny) = (flow.nx, flow.ny);
        let mut pixels = Vec::with_capacity(4 * nx * ny);
        if self.kind == Kind::Convection {
            // Blue (cold) to white to red (warm).
            for t in &flow.fields.temperature {
                let s = (2.0 * t).clamp(-1.0, 1.0) as f32;
                let (r, g, b) = if s > 0.0 { (1.0, 1.0 - 0.75 * s, 1.0 - 0.9 * s) } else { (1.0 + 0.85 * s, 1.0 + 0.6 * s, 1.0) };
                let dim = 0.9;
                pixels.extend_from_slice(&[(r * dim * 255.0) as u8, (g * dim * 255.0) as u8, (b * dim * 255.0) as u8, 255]);
            }
        } else {
            // Vorticity scaled by a typical value of the scene.
            let scale = if self.kind == Kind::Cavity { 10.0 } else { 1.0 };
            let dim = if self.kind == Kind::Cavity { 0.7 } else { 0.35 };
            for w in flow.vorticity() {
                let s = (w / scale).clamp(-1.0, 1.0);
                let brightness = (s.abs().sqrt() * dim) as f32;
                let (r, g, b) = if s > 0.0 { (1.0, 0.55, 0.1) } else { (0.15, 0.45, 1.0) };
                pixels.extend_from_slice(&[(r * brightness * 255.0) as u8, (g * brightness * 255.0) as u8, (b * brightness * 255.0) as u8, 255]);
            }
        }
        let texture = self.background.get_or_init(|| {
            let texture = Texture2D::from_rgba8(nx as u16, ny as u16, &pixels);
            texture.set_filter(FilterMode::Linear);
            texture
        });
        texture.update_from_bytes(nx as u32, ny as u32, &pixels);
        let rect = self.box_rect();
        draw_texture_ex(texture, rect.x, rect.y, WHITE, DrawTextureParams { dest_size: Some(vec2(rect.w, rect.h)), ..Default::default() });
    }

    /// The tracers, splatted into a texture four times finer than the grid.
    fn draw_tracers(&self) {
        if self.tracers.positions.is_empty() {
            return;
        }
        let flow = &self.flow;
        let (w, h) = (4 * flow.nx, 4 * flow.ny);
        let mut pixels = vec![0u8; 4 * w * h];
        let scale = 4.0 / flow.h;
        for (p, color) in self.tracers.positions.iter().zip(&self.tracers.colors) {
            let (i, j) = (((p.0 * scale) as usize).min(w - 1), ((p.1 * scale) as usize).min(h - 1));
            let k = 4 * (j * w + i);
            pixels[k..k + 3].copy_from_slice(color);
            pixels[k + 3] = 255;
        }
        let texture = self.dots.get_or_init(|| {
            let texture = Texture2D::from_rgba8(w as u16, h as u16, &pixels);
            texture.set_filter(FilterMode::Nearest);
            texture
        });
        texture.update_from_bytes(w as u32, h as u32, &pixels);
        let rect = self.box_rect();
        draw_texture_ex(texture, rect.x, rect.y, WHITE, DrawTextureParams { dest_size: Some(vec2(rect.w, rect.h)), ..Default::default() });
    }

    /// Velocity profiles through the centre of the cavity, with Ghia's points: u(y) along the
    /// vertical line (drawn at x = 0.5 + 0.3 u) and v(x) along the horizontal one.
    fn draw_cavity_profiles(&self) {
        let column = self.ghia_index();
        let pick = |row: &(f64, f64, f64)| (if column == 0 { row.1 } else { row.2 }) as f32;
        let (gain, width) = (0.3, 0.004);
        let points = 200;
        for k in 0..points {
            let (s0, s1) = (k as f64 / points as f64, (k + 1) as f64 / points as f64);
            let (u0, u1) = (self.flow.velocity(0.5, s0).0 as f32, self.flow.velocity(0.5, s1).0 as f32);
            draw_line(0.5 + gain * u0, s0 as f32, 0.5 + gain * u1, s1 as f32, width, YELLOW);
            let (v0, v1) = (self.flow.velocity(s0, 0.5).1 as f32, self.flow.velocity(s1, 0.5).1 as f32);
            draw_line(s0 as f32, 0.5 + gain * v0, s1 as f32, 0.5 + gain * v1, width, GREEN);
        }
        for row in &GHIA_U {
            draw_circle(0.5 + gain * pick(row), row.0 as f32, 0.009, WHITE);
        }
        for row in &GHIA_V {
            draw_circle(row.0 as f32, 0.5 + gain * pick(row), 0.009, WHITE);
        }
        draw_line(0.5, 0.0, 0.5, 1.0, 0.002, GRAY);
        draw_line(0.0, 0.5, 1.0, 0.5, 0.002, GRAY);
    }
}

impl Simulation for NavierStokes {
    fn step(&mut self, _dt: f64) {
        // As in the vortex street: drop steps rather than the frame rate on a slow machine.
        if self.steps_this_frame >= MAX_STEPS_PER_FRAME {
            self.capped = true;
            return;
        }
        self.steps_this_frame += 1;
        self.flow.step();
        self.tracers.advance(&self.flow, self.flow.dt);
        if self.kind == Kind::ShearLayer {
            self.growth_samples.push((self.flow.t, mode_amplitude(&self.flow, 2).ln()));
            let t = self.flow.t;
            self.growth_samples.retain(|(s, _)| t - s <= 1.5);
        }
    }

    fn is_over(&self) -> bool {
        false
    }

    fn dt(&self) -> f64 {
        self.flow.dt
    }

    /// Time units per second of real time: set so each scene's motion is easy to follow.
    fn time_scale(&self) -> f64 {
        let base = match self.kind {
            Kind::Cavity => 0.5,
            Kind::ShearLayer => 4.0,
            Kind::Convection => 1.0,
        };
        base * self.speed
    }

    fn world_bounds(&self) -> Rect {
        // The box below the HUD lines (the top third of the window), centred, at 3:2.
        match self.kind {
            Kind::Cavity => Rect::new(-0.6625, -0.03, 2.325, 1.55),
            Kind::ShearLayer => {
                let rect = self.box_rect();
                let height = (rect.h + 0.3) / 0.67;
                Rect::new(rect.w / 2.0 - 0.75 * height, rect.y - 0.3, 1.5 * height, height)
            }
            Kind::Convection => Rect::new(-0.05, -0.05, 4.1, 2.733),
        }
    }

    fn handle_input(&mut self) {
        self.steps_this_frame = 0;
        self.capped = false;
        let speed = self.speed;
        for (key, kind) in [(KeyCode::Key1, Kind::Cavity), (KeyCode::Key2, Kind::ShearLayer), (KeyCode::Key3, Kind::Convection)] {
            if is_key_pressed(key) {
                *self = Self::scene(kind, REYNOLDS_CAVITY[0]);
                self.speed = speed;
            }
        }
        if self.kind == Kind::Cavity && is_key_pressed(KeyCode::R) {
            let other = if self.reynolds == REYNOLDS_CAVITY[0] { REYNOLDS_CAVITY[1] } else { REYNOLDS_CAVITY[0] };
            *self = Self::scene(Kind::Cavity, other);
            self.speed = speed;
        }
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
    }

    fn draw(&self) {
        self.draw_background();
        self.draw_tracers();
        let rect = self.box_rect();
        if self.kind == Kind::Cavity {
            self.draw_cavity_profiles();
            // The lid.
            draw_rectangle(rect.x - 0.01, rect.y + rect.h, rect.w + 0.02, 0.012, LIGHTGRAY);
        }
        draw_rectangle_lines(rect.x, rect.y, rect.w, rect.h, 0.004 * rect.w.max(rect.h), GRAY);
    }

    fn hud(&self) -> Vec<String> {
        let flow = &self.flow;
        let mut lines = vec![format!(
            "[1] cavity [2] shear [3] convection  t = {:.1}  {}x (UP/DOWN){}",
            flow.t,
            self.speed,
            if self.capped { " - capped" } else { "" }
        )];
        match self.kind {
            Kind::Cavity => {
                let ghia = GHIA_VORTEX[self.ghia_index()];
                let (psi, x, y) = self.primary_vortex();
                let (du, dv) = self.ghia_deviation();
                lines.push(format!("Re = {} (R)  {}x{}  e.g. 1 cm of water, lid {} cm/s", self.reynolds, flow.nx, flow.ny, self.reynolds / 100.0));
                lines.push(format!("eddy psi {:.4} ({:.3}, {:.3})  Ghia {:.4} ({:.3}, {:.3})", psi, x, y, ghia.0, ghia.1, ghia.2));
                lines.push(format!("vs Ghia (dots): |du| <= {:.3} yellow, |dv| <= {:.3} green", du, dv));
            }
            Kind::ShearLayer => {
                lines.push(format!("Re = {}   u = tanh(y)   two {:.2}-long waves", SHEAR_REYNOLDS, TAU / SHEAR_WAVE_NUMBER));
                let amplitude = mode_amplitude(flow, 2);
                // The linear theory only holds while the wave is small.
                let rate = match self.growth_rate() {
                    Some(_) if amplitude > SHEAR_LINEAR_LIMIT => "nonlinear".to_string(),
                    Some(rate) => format!("{rate:.3}"),
                    None => "-".to_string(),
                };
                lines.push(format!("amplitude {:.1e}  growth {} (inviscid theory {})", amplitude, rate, SHEAR_GROWTH_INVISCID));
            }
            Kind::Convection => {
                lines.push(format!("Ra = {:.0e}   Pr = {} (water)   {}x{} cells", CONVECTION_RAYLEIGH, CONVECTION_PRANDTL, flow.nx, flow.ny));
                lines.push(format!("Nusselt {:.2} (1 = conduction only)  onset at Ra 1708", flow.nusselt()));
            }
        }
        lines.push(format!("kinetic energy {:.4}   max |div u| {:.0e}", flow.kinetic_energy(), flow.max_divergence()));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_solver_inverts_the_laplacian() {
        for periodic in [false, true] {
            let (nx, ny) = (16, 12);
            let rng = RandGenerator::new();
            rng.srand(1);
            let phi: Vec<f64> = (0..nx * ny).map(|_| rng.gen_range(-1.0, 1.0)).collect();
            let at = |i: isize, j: isize| {
                let j = j.clamp(0, ny as isize - 1);
                let i = if periodic { i.rem_euclid(nx as isize) } else { i.clamp(0, nx as isize - 1) };
                phi[j as usize * nx + i as usize]
            };
            let mut rhs = vec![0.0; nx * ny];
            for j in 0..ny as isize {
                for i in 0..nx as isize {
                    rhs[j as usize * nx + i as usize] = at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) - 4.0 * at(i, j);
                }
            }
            let solved = Poisson::new(nx, ny, periodic).solve(&rhs);
            // phi is only defined up to a constant.
            let shift = solved[0] - phi[0];
            let worst = solved.iter().zip(&phi).map(|(s, p)| (s - shift - p).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-11, "periodic {periodic}: off by {worst}");
        }
    }

    #[test]
    fn projection_leaves_no_divergence() {
        let mut flow = cavity(100.0);
        let rng = RandGenerator::new();
        rng.srand(2);
        flow.fields.u.iter_mut().for_each(|u| *u = rng.gen_range(-1.0, 1.0));
        flow.fields.v.iter_mut().for_each(|v| *v = rng.gen_range(-1.0, 1.0));
        let n = flow.nx;
        for j in 0..n {
            flow.fields.u[j * (n + 1)] = 0.0;
            flow.fields.u[j * (n + 1) + n] = 0.0;
        }
        for i in 0..n {
            flow.fields.v[i] = 0.0;
            flow.fields.v[n * n + i] = 0.0;
        }
        let mut fields = flow.fields.clone();
        flow.project(&mut fields);
        flow.fields = fields;
        assert!(flow.max_divergence() < 1e-10, "max divergence {}", flow.max_divergence());
    }

    #[test]
    fn taylor_green_vortices_decay_at_the_viscous_rate() {
        // u = sin x cos y, v = -cos x sin y in a periodic 2 pi box (free-slip walls at y = 0 and
        // 2 pi are exact for it), decaying as exp(-2 nu t). Nonlinear terms cancel exactly.
        let n = 64;
        let h = TAU / n as f64;
        let nu = 0.05;
        let mut flow = Flow::new(n, n, h, true, true, nu, 0.2 * h);
        for j in 0..n {
            for i in 0..=n {
                let (x, y) = (i as f64 * h, (j as f64 + 0.5) * h);
                flow.fields.u[j * (n + 1) + i] = x.sin() * y.cos();
            }
        }
        for j in 0..=n {
            for i in 0..n {
                let (x, y) = ((i as f64 + 0.5) * h, j as f64 * h);
                flow.fields.v[j * n + i] = -x.cos() * y.sin();
            }
        }
        let energy0 = flow.kinetic_energy();
        let steps = (2.0 / flow.dt).round() as usize;
        for _ in 0..steps {
            flow.step();
        }
        // The discrete Laplacian of sin(x) is -(2 - 2 cos h) / h^2 sin(x), slightly less than -1.
        let decay = 2.0 * (2.0 - 2.0 * h.cos()) / (h * h);
        let expected = energy0 * (-2.0 * nu * decay * flow.t).exp();
        let error = flow.kinetic_energy() / expected - 1.0;
        assert!(error.abs() < 1e-6, "energy off by {error}");
        let continuum = energy0 * (-4.0 * nu * flow.t).exp();
        assert!((flow.kinetic_energy() / continuum - 1.0).abs() < 2e-3);
    }

    #[test]
    fn energy_is_conserved_without_viscosity() {
        // The advection conserves energy exactly; only the time stepping's error is left, and it
        // shrinks 8 times when the step is halved (third order).
        let change = |scale: f64| {
            let mut flow = shear_layer(64, f64::INFINITY, 0.05, 0.01);
            flow.dt *= scale;
            let energy0 = flow.kinetic_energy();
            let steps = (200.0 / scale).round() as usize;
            for _ in 0..steps {
                flow.step();
            }
            flow.kinetic_energy() / energy0 - 1.0
        };
        let (coarse, fine) = (change(1.0), change(0.5));
        assert!(coarse.abs() < 1e-6, "energy changed by {coarse}");
        let ratio = coarse / fine;
        assert!(ratio > 7.0 && ratio < 9.0, "error ratio {ratio}");
    }

    #[test]
    fn cavity_at_reynolds_100_matches_ghia() {
        let mut sim = NavierStokes::scene(Kind::Cavity, 100.0);
        while sim.flow.t < 25.0 {
            sim.flow.step();
        }
        let (du, dv) = sim.ghia_deviation();
        assert!(du < 0.01 && dv < 0.01, "worst deviation u {du}, v {dv}");
        let (psi, x, y) = sim.primary_vortex();
        let (ghia_psi, ghia_x, ghia_y) = GHIA_VORTEX[0];
        assert!((psi / ghia_psi - 1.0).abs() < 0.01, "psi {psi}");
        assert!((x - ghia_x).abs() < 0.02 && (y - ghia_y).abs() < 0.02, "centre ({x}, {y})");
    }

    #[test]
    fn shear_layer_grows_at_the_linear_rate() {
        // Nearly inviscid (Re 1e4), fitted over t = 15..30: after the start-up transient, well
        // before saturation. The rate converges at second order in the grid spacing, so
        // Richardson extrapolation of the two grids estimates the converged rate.
        let rate = |nx: usize| {
            let mut flow = shear_layer(nx, 1e4, SHEAR_KICK, 0.0);
            let mut samples = Vec::new();
            while flow.t < 30.0 {
                flow.step();
                if flow.t >= 15.0 {
                    samples.push((flow.t, mode_amplitude(&flow, 2).ln()));
                }
            }
            let ((t0, a0), (t1, a1)) = (samples[0], samples[samples.len() - 1]);
            (a1 - a0) / (t1 - t0)
        };
        let (coarse, fine) = (rate(128), rate(256));
        let converged = fine + (fine - coarse) / 3.0;
        assert!((fine / SHEAR_GROWTH_INVISCID - 1.0).abs() < 0.02, "growth rate {fine} on 256 cells");
        assert!((converged / SHEAR_GROWTH_INVISCID - 1.0).abs() < 0.01, "extrapolated growth rate {converged}");
    }

    #[test]
    fn convection_starts_above_the_critical_rayleigh_number() {
        // Two deep, close to the critical wavelength 2 pi / 3.117 = 2.016 (Ra_c is 1709 at
        // wave number pi instead of 1708): below Ra_c the noise dies out, above it grows.
        let rate = |rayleigh: f64| {
            let mut flow = convection(32, 16, rayleigh, 1.0, 3);
            let energy = |f: &Flow| f.kinetic_energy().ln();
            let mut samples = Vec::new();
            while flow.t < 60.0 {
                flow.step();
                samples.push((flow.t, energy(&flow)));
            }
            let (t0, e0) = samples[samples.len() / 2];
            let (t1, e1) = samples[samples.len() - 1];
            (e1 - e0) / (t1 - t0)
        };
        let (below, above) = (rate(1550.0), rate(1900.0));
        assert!(below < 0.0, "grows at Ra 1550: {below}");
        assert!(above > 0.0, "decays at Ra 1900: {above}");
    }
}

