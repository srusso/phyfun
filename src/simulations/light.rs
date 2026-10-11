use std::cell::OnceCell;
use std::f64::consts::{PI, TAU};

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: length in micrometres (um), time in femtoseconds (fs). The magnetic field is stored
// multiplied by the impedance of free space, so E and H have the same units and that constant
// never appears. Fields are for light polarised with E along z (2D: nothing varies along z).

const C: f64 = 0.299_792_458; // um/fs, exact
/// Vacuum wavelength of the He-Ne laser line.
const WAVELENGTH: f64 = 0.6328; // um
const PERIOD: f64 = WAVELENGTH / C; // 2.11 fs
const OMEGA: f64 = TAU / PERIOD;
/// Refractive index of Schott N-BK7 at 632.8 nm, from Schott's Sellmeier formula (1.515089).
const N_BK7: f64 = 1.51509;

const CELLS_PER_WAVELENGTH: f64 = 32.0;
const STEPS_PER_PERIOD: f64 = 50.0;
const CELL: f64 = WAVELENGTH / CELLS_PER_WAVELENGTH;
const DT: f64 = PERIOD / STEPS_PER_PERIOD;
/// Courant number, 0.64. The 2D stability limit is 1/sqrt(2) = 0.707.
const S: f64 = C * DT / CELL;

/// Cells in the absorbing layer on each side, and in the window inside it.
const PML: usize = 12;
const INTERIOR_X: usize = 900;
const INTERIOR_Y: usize = 600;
const NX: usize = INTERIOR_X + 2 * PML;
const NY: usize = INTERIOR_Y + 2 * PML;
/// Grading of the layer: sigma dt = SIGMA_MAX (d / PML)^3, with SIGMA_MAX = 0.8 (m + 1) S for m = 3.
const SIGMA_MAX: f64 = 2.048;

// Geometry of the scenes, um.
const SOURCE_COLUMN: usize = PML + 30;
const BEAM_WAIST: f64 = 2.0;
const AXIS_Y: f64 = 300.0 * CELL; // 5.9325 um, the middle of the window
const PIVOT: (f64, f64) = (7.0, AXIS_Y);
const LENS_RADIUS: f64 = 4.0;
const LENS_FLAT_FACE: f64 = 4.5;

/// Steps between updates of the readouts (4 periods).
const REPORT_STEPS: u32 = 200;
const START_SPEED: f64 = 15.0; // fs per second of real time: 6 steps per frame
const MIN_SPEED: f64 = 1.875;
const MAX_SPEED: f64 = 60.0;

/// The incident wave: a continuous sine with a slow start, or (in tests) a Gaussian pulse.
#[derive(Clone, Copy)]
enum Signal {
    Continuous,
    /// Only the tests use pulses.
    #[cfg_attr(not(test), allow(dead_code))]
    Pulse { tau: f64, t0: f64 },
}

impl Signal {
    fn value(&self, t: f64) -> f64 {
        match *self {
            Signal::Continuous => {
                let ramp = if t < 5.0 * PERIOD { (PI * t / (10.0 * PERIOD)).sin().powi(2) } else { 1.0 };
                ramp * (OMEGA * t).sin()
            }
            Signal::Pulse { tau, t0 } => {
                if t > 2.0 * t0 { 0.0 } else { (-((t - t0) / tau).powi(2)).exp() * (OMEGA * (t - t0)).sin() }
            }
        }
    }
}

/// A one-way line source on one column (a total-field / scattered-field boundary) in a region of
/// uniform index `index`: the beam leaves towards +x only, and everything to the left of the
/// column is light coming back.
struct Beam {
    column: usize,
    /// Amplitude across the beam, per row.
    profile: Vec<f32>,
    index: f64,
    signal: Signal,
}

impl Beam {
    /// Time the grid's own wave takes to cross half a cell, which separates the electric and
    /// the magnetic corrections.
    fn delay(&self) -> f64 {
        (self.index * (PI / STEPS_PER_PERIOD).sin() / S).asin() / OMEGA
    }
}

/// Yee's finite-difference time-domain scheme (1966): Ez at cell centres and whole time steps,
/// Hx and Hy on cell edges and half steps, centred differences. Second order, explicit; in a
/// closed lossless box it conserves a discrete energy exactly. The outer walls are perfect
/// magnetic mirrors. A convolutional perfectly matched layer (CPML, Roden and Gedney 2000)
/// absorbs outgoing waves. No macroquad calls.
struct Grid {
    nx: usize,
    ny: usize,
    pml_x: usize,
    pml_y: usize,
    /// `ez[j * nx + i]` at the centre of cell (i, j).
    ez: Vec<f32>,
    /// `hx[j * nx + i]` on the edge below cell (i, j); rows 0 and ny are walls.
    hx: Vec<f32>,
    /// `hy[j * (nx + 1) + i]` on the edge left of cell (i, j); columns 0 and nx are walls.
    hy: Vec<f32>,
    /// S / n^2 per cell.
    ce: Vec<f32>,
    /// CPML coefficients b = exp(-sigma dt) and a = b - 1, at cell centres ("e") and at edges
    /// ("h"), per column and per row.
    bx_e: Vec<f32>,
    ax_e: Vec<f32>,
    bx_h: Vec<f32>,
    ax_h: Vec<f32>,
    by_e: Vec<f32>,
    ay_e: Vec<f32>,
    by_h: Vec<f32>,
    ay_h: Vec<f32>,
    /// Memory of the CPML for each of the four differences.
    psi_hx: Vec<f32>,
    psi_hy: Vec<f32>,
    psi_ez_hy: Vec<f32>,
    psi_ez_hx: Vec<f32>,
    step_count: u64,
}

/// CPML coefficients along an axis of `n` cells with a layer of `layer` cells on each side.
/// Returns (b, a) for cell centres and (b, a) for edges (n + 1 of them).
fn layer_tables(n: usize, layer: usize) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
    let coefficients = |depth: f64| {
        let sigma_dt = if layer == 0 { 0.0 } else { SIGMA_MAX * (depth / layer as f64).powi(3) };
        let b = (-sigma_dt).exp();
        (b as f32, (b - 1.0) as f32)
    };
    let (l, nf) = (layer as f64, n as f64);
    let centres = (0..n).map(|i| {
        let i = i as f64;
        coefficients(if i < l { l - i - 0.5 } else if i >= nf - l { i - (nf - l) + 0.5 } else { 0.0 })
    });
    let edges = (0..=n).map(|i| {
        let i = i as f64;
        coefficients(if i <= l { l - i } else if i >= nf - l { i - (nf - l) } else { 0.0 })
    });
    (centres.unzip(), edges.unzip())
}

impl Grid {
    fn new(nx: usize, ny: usize, pml_x: usize, pml_y: usize) -> Self {
        let ((bx_e, ax_e), (bx_h, ax_h)) = layer_tables(nx, pml_x);
        let ((by_e, ay_e), (by_h, ay_h)) = layer_tables(ny, pml_y);
        Self {
            nx,
            ny,
            pml_x,
            pml_y,
            ez: vec![0.0; nx * ny],
            hx: vec![0.0; nx * (ny + 1)],
            hy: vec![0.0; (nx + 1) * ny],
            ce: vec![S as f32; nx * ny],
            bx_e,
            ax_e,
            bx_h,
            ax_h,
            by_e,
            ay_e,
            by_h,
            ay_h,
            psi_hx: vec![0.0; nx * (ny + 1)],
            psi_hy: vec![0.0; (nx + 1) * ny],
            psi_ez_hy: vec![0.0; nx * ny],
            psi_ez_hx: vec![0.0; nx * ny],
            step_count: 0,
        }
    }

    /// World coordinates (um) of the centre of cell (i, j); the window inside the layer starts at 0.
    fn x(&self, i: usize) -> f64 {
        (i as f64 - self.pml_x as f64 + 0.5) * CELL
    }

    fn y(&self, j: usize) -> f64 {
        (j as f64 - self.pml_y as f64 + 0.5) * CELL
    }

    /// Sets the refractive index from a function of position. A cell's n^2 is the mean over 4 x 4
    /// points inside it: for E parallel to the surface, the arithmetic mean of n^2 is the right
    /// effective value, so glass edges need no staircase.
    fn set_index(&mut self, index: impl Fn(f64, f64) -> f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let (cx, cy) = (self.x(i), self.y(j));
                let mut sum = 0.0;
                for a in 0..4 {
                    for b in 0..4 {
                        let offset = |k: usize| ((k as f64 + 0.5) / 4.0 - 0.5) * CELL;
                        sum += index(cx + offset(a), cy + offset(b)).powi(2);
                    }
                }
                self.ce[j * self.nx + i] = (S / (sum / 16.0)) as f32;
            }
        }
    }

    /// n^2 of a cell (tests only).
    #[cfg(test)]
    fn n2(&self, i: usize, j: usize) -> f64 {
        S / self.ce[j * self.nx + i] as f64
    }

    /// Hx and Hy from the curl of E: half a step.
    fn step_h(&mut self, beam: Option<&Beam>) {
        let (nx, ny) = (self.nx, self.ny);
        let s = S as f32;
        for j in 1..ny {
            let (above, below) = (&self.ez[j * nx..(j + 1) * nx], &self.ez[(j - 1) * nx..j * nx]);
            for ((h, a), b) in self.hx[j * nx..(j + 1) * nx].iter_mut().zip(above).zip(below) {
                *h -= s * (a - b);
            }
        }
        for j in 0..ny {
            let row = &self.ez[j * nx..(j + 1) * nx];
            let hy = &mut self.hy[j * (nx + 1)..(j + 1) * (nx + 1)];
            for ((h, right), left) in hy[1..nx].iter_mut().zip(&row[1..]).zip(&row[..nx - 1]) {
                *h += s * (right - left);
            }
        }

        // Absorbing layer: the extra term S psi, where psi remembers the difference.
        for j in 1..ny {
            let a = self.ay_h[j];
            if a != 0.0 {
                let b = self.by_h[j];
                for i in 0..nx {
                    let d = self.ez[j * nx + i] - self.ez[(j - 1) * nx + i];
                    let psi = &mut self.psi_hx[j * nx + i];
                    *psi = b * *psi + a * d;
                    self.hx[j * nx + i] -= s * *psi;
                }
            }
        }
        for i in (1..nx).filter(|&i| self.ax_h[i] != 0.0) {
            let (a, b) = (self.ax_h[i], self.bx_h[i]);
            for j in 0..ny {
                let d = self.ez[j * nx + i] - self.ez[j * nx + i - 1];
                let psi = &mut self.psi_hy[j * (nx + 1) + i];
                *psi = b * *psi + a * d;
                self.hy[j * (nx + 1) + i] += s * *psi;
            }
        }

        if let Some(beam) = beam {
            let t = self.step_count as f64 * DT;
            let g = beam.signal.value(t);
            for j in 0..ny {
                self.hy[j * (nx + 1) + beam.column] -= (S * beam.profile[j] as f64 * g) as f32;
            }
        }
    }

    /// Ez from the curl of H: half a step, so that E is centred on H in time.
    fn step_e(&mut self, beam: Option<&Beam>) {
        let (nx, ny) = (self.nx, self.ny);
        for j in 0..ny {
            let hy = &self.hy[j * (nx + 1)..(j + 1) * (nx + 1)];
            let (hx_above, hx_below) = (&self.hx[(j + 1) * nx..(j + 2) * nx], &self.hx[j * nx..(j + 1) * nx]);
            let ce = &self.ce[j * nx..(j + 1) * nx];
            let ez = &mut self.ez[j * nx..(j + 1) * nx];
            // Slices of the same length, so the loop needs no bounds checks.
            let (hy_right, hy_left) = (&hy[1..=nx], &hy[..nx]);
            for i in 0..nx {
                ez[i] += ce[i] * ((hy_right[i] - hy_left[i]) - (hx_above[i] - hx_below[i]));
            }
        }

        for i in (0..nx).filter(|&i| self.ax_e[i] != 0.0) {
            let (a, b) = (self.ax_e[i], self.bx_e[i]);
            for j in 0..ny {
                let d = self.hy[j * (nx + 1) + i + 1] - self.hy[j * (nx + 1) + i];
                let psi = &mut self.psi_ez_hy[j * nx + i];
                *psi = b * *psi + a * d;
                self.ez[j * nx + i] += self.ce[j * nx + i] * *psi;
            }
        }
        for j in (0..ny).filter(|&j| self.ay_e[j] != 0.0) {
            let (a, b) = (self.ay_e[j], self.by_e[j]);
            for i in 0..nx {
                let d = self.hx[(j + 1) * nx + i] - self.hx[j * nx + i];
                let psi = &mut self.psi_ez_hx[j * nx + i];
                *psi = b * *psi + a * d;
                self.ez[j * nx + i] -= self.ce[j * nx + i] * *psi;
            }
        }

        if let Some(beam) = beam {
            let t = self.step_count as f64 * DT;
            let g = beam.signal.value(t + DT / 2.0 + beam.delay());
            for j in 0..ny {
                let k = j * nx + beam.column;
                self.ez[k] += (self.ce[k] as f64 * beam.index * beam.profile[j] as f64 * g) as f32;
            }
        }
        self.step_count += 1;
    }

    /// Energy flowing in +x through the edge at column `i` of row `j`: the Poynting vector's x
    /// component, -Ez Hy, with E averaged over the two cells that share the edge.
    fn flux_x(&self, i: usize, j: usize) -> f64 {
        -(self.hy[j * (self.nx + 1) + i] as f64) * (self.ez[j * self.nx + i - 1] + self.ez[j * self.nx + i]) as f64 / 2.0
    }

    /// Energy flowing in +y through the edge below cell (i, j): +Ez Hx.
    fn flux_y(&self, i: usize, j: usize) -> f64 {
        self.hx[j * self.nx + i] as f64 * (self.ez[(j - 1) * self.nx + i] + self.ez[j * self.nx + i]) as f64 / 2.0
    }

    /// Ez at a world position, interpolated bilinearly between cell centres (tests only).
    #[cfg(test)]
    fn ez_at(&self, x: f64, y: f64) -> f64 {
        let (fx, fy) = (x / CELL + self.pml_x as f64 - 0.5, y / CELL + self.pml_y as f64 - 0.5);
        let (i, j) = (fx.floor() as usize, fy.floor() as usize);
        let (u, v) = (fx - i as f64, fy - j as f64);
        let at = |i: usize, j: usize| self.ez[j * self.nx + i] as f64;
        at(i, j) * (1.0 - u) * (1.0 - v) + at(i + 1, j) * u * (1.0 - v) + at(i, j + 1) * (1.0 - u) * v + at(i + 1, j + 1) * u * v
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Scene {
    /// Vacuum to glass.
    Refraction,
    /// Glass to vacuum: total internal reflection and its evanescent wave.
    TotalReflection,
    /// Glass, a vacuum gap, glass.
    Tunnelling,
    /// A plano-convex cylinder lens.
    Lens,
}

impl Scene {
    /// Refractive index at a world position. `angle` is the angle of incidence (radians) and
    /// `gap` the width of the vacuum gap (um).
    fn index(self, angle: f64, gap: f64, x: f64, y: f64) -> f64 {
        // Distance along the surface normal from the pivot.
        let s = (x - PIVOT.0) * angle.cos() + (y - PIVOT.1) * angle.sin();
        let glass = match self {
            Scene::Refraction => s >= 0.0,
            Scene::TotalReflection => s < 0.0,
            Scene::Tunnelling => s < 0.0 || s >= gap,
            Scene::Lens => (x - PIVOT.0).powi(2) + (y - PIVOT.1).powi(2) <= LENS_RADIUS * LENS_RADIUS && x <= LENS_FLAT_FACE,
        };
        if glass { N_BK7 } else { 1.0 }
    }

    /// Index at the source column, where the beam starts.
    fn source_index(self) -> f64 {
        match self {
            Scene::Refraction | Scene::Lens => 1.0,
            Scene::TotalReflection | Scene::Tunnelling => N_BK7,
        }
    }

    fn start_angle(self) -> f64 {
        45.0
    }
}

/// What the meters have measured over a window.
#[derive(Clone, Default)]
struct Report {
    /// Power the source injects.
    source: f64,
    reflected: f64,
    transmitted: f64,
    /// Refraction angle in degrees, from where the transmitted light leaves the window.
    refraction_deg: Option<f64>,
    /// Mean Ez^2 over the window along the two rows next to the axis, per grid column.
    axis_intensity: Vec<f64>,
}

impl Report {
    fn reflectance(&self) -> f64 {
        self.reflected / self.source
    }

    fn transmittance(&self) -> f64 {
        self.transmitted / self.source
    }
}

/// Flux and intensity sums, added to after every half step.
struct Meters {
    source: f64,
    reflected: f64,
    transmitted: f64,
    /// Transmitted flux times position.
    moment: (f64, f64),
    axis_intensity: Vec<f64>,
    samples: u32,
}

impl Meters {
    fn new(nx: usize) -> Self {
        Self { source: 0.0, reflected: 0.0, transmitted: 0.0, moment: (0.0, 0.0), axis_intensity: vec![0.0; nx], samples: 0 }
    }
}

/// A grid with its source and scene, and the meters (no window needed, so tests can use it).
struct Bench {
    grid: Grid,
    beam: Beam,
    scene: Scene,
    /// Angle of incidence in degrees, and the vacuum gap in um.
    angle: f64,
    gap: f64,
    meters: Meters,
    window: u32,
    report: Option<Report>,
}

impl Bench {
    fn new(scene: Scene, angle: f64, gap: f64, signal: Signal) -> Self {
        let mut grid = Grid::new(NX, NY, PML, PML);
        grid.set_index(|x, y| scene.index(angle.to_radians(), gap, x, y));
        let profile = (0..NY).map(|j| (-((grid.y(j) - AXIS_Y) / BEAM_WAIST).powi(2)).exp() as f32).collect();
        let beam = Beam { column: SOURCE_COLUMN, profile, index: scene.source_index(), signal };
        Self { grid, beam, scene, angle, gap, meters: Meters::new(NX), window: 0, report: None }
    }

    /// Flux of the source, and of the light leaving a rectangle 2 cells inside the layer, split
    /// into reflected and transmitted by which side of the surface it leaves on. Called after
    /// each half step, so that E and H are both used at about the same time.
    fn measure_fluxes(&mut self) {
        let grid = &self.grid;
        let s_of = |x: f64, y: f64| (x - PIVOT.0) * self.angle.to_radians().cos() + (y - PIVOT.1) * self.angle.to_radians().sin();
        let m = &mut self.meters;
        let c = SOURCE_COLUMN;
        for j in 0..grid.ny {
            m.source += grid.flux_x(c + 3, j) - grid.flux_x(c - 3, j);
        }
        let (left, right, bottom, top) = (PML + 2, NX - PML - 2, PML + 2, NY - PML - 2);
        let add = |m: &mut Meters, outward: f64, x: f64, y: f64| {
            if s_of(x, y) >= self.gap / 2.0 {
                m.transmitted += outward;
                m.moment.0 += outward.max(0.0) * x;
                m.moment.1 += outward.max(0.0) * y;
            } else {
                m.reflected += outward;
            }
        };
        for j in bottom..top {
            let y = grid.y(j);
            add(m, -grid.flux_x(left, j), (left - PML) as f64 * CELL, y);
            add(m, grid.flux_x(right, j), (right - PML) as f64 * CELL, y);
        }
        for i in left..right {
            let x = grid.x(i);
            add(m, -grid.flux_y(i, bottom), x, (bottom - PML) as f64 * CELL);
            add(m, grid.flux_y(i, top), x, (top - PML) as f64 * CELL);
        }
    }

    fn step(&mut self) {
        self.grid.step_h(Some(&self.beam));
        self.measure_fluxes();
        self.grid.step_e(Some(&self.beam));
        self.measure_fluxes();

        // Mean Ez^2 along the two rows next to the beam's axis.
        let (nx, row) = (self.grid.nx, PML + 299);
        for i in 0..nx {
            let e = (self.grid.ez[row * nx + i] as f64).powi(2) + (self.grid.ez[(row + 1) * nx + i] as f64).powi(2);
            self.meters.axis_intensity[i] += e / 2.0;
        }
        self.meters.samples += 1;

        self.window += 1;
        if self.window == REPORT_STEPS {
            self.report = Some(self.read_meters());
            self.reset_meters();
        }
    }

    /// The meters' totals as a report. Sums are over half steps, so the common factor of 2 in
    /// the fluxes cancels in the ratios.
    fn read_meters(&self) -> Report {
        let m = &self.meters;
        let refraction_deg = (m.transmitted > 0.0 && m.moment.0 + m.moment.1 != 0.0).then(|| {
            let (q_x, q_y) = (m.moment.0 / m.transmitted, m.moment.1 / m.transmitted);
            self.angle - (q_y - PIVOT.1).atan2(q_x - PIVOT.0).to_degrees()
        });
        Report {
            source: m.source,
            reflected: m.reflected,
            transmitted: m.transmitted,
            refraction_deg,
            axis_intensity: m.axis_intensity.iter().map(|sum| sum / m.samples.max(1) as f64).collect(),
        }
    }

    fn reset_meters(&mut self) {
        self.meters = Meters::new(NX);
        self.window = 0;
    }

    /// Runs `steps` steps with fresh meters and reads them.
    fn measure(&mut self, steps: usize) -> Report {
        self.reset_meters();
        for _ in 0..steps {
            self.step();
        }
        self.read_meters()
    }
}

/// Reflectance of E polarised along the surface (s polarisation): Fresnel's formula, and 1 beyond
/// the critical angle. `angle` is the angle of incidence in radians.
fn fresnel_reflectance(n1: f64, n2: f64, angle: f64) -> f64 {
    let sin_r = n1 * angle.sin() / n2;
    if sin_r > 1.0 {
        return 1.0;
    }
    let (a, b) = (n1 * angle.cos(), n2 * (1.0 - sin_r * sin_r).sqrt());
    ((a - b) / (a + b)).powi(2)
}

/// Fraction of the power that tunnels through a vacuum gap `d` between two blocks of glass.
fn tunnelling_transmittance(angle: f64, d: f64) -> f64 {
    let k = TAU / WAVELENGTH;
    let kx = k * N_BK7 * angle.cos();
    let kappa = k * (N_BK7 * N_BK7 * angle.sin().powi(2) - 1.0).sqrt();
    let a = ((kx * kx + kappa * kappa) / (2.0 * kx * kappa)).powi(2);
    1.0 / (1.0 + a * (kappa * d).sinh().powi(2))
}

/// Light, drawn as the actual oscillating electric field, in a window 18 x 12 micrometres: a red
/// He-Ne laser beam (632.8 nm) meets glass (Schott N-BK7). The only physics is Maxwell's two curl
/// equations on a grid (Yee's FDTD scheme, 32 cells per wavelength) and a map of the refractive
/// index; refraction, partial reflection, total internal reflection with its evanescent wave,
/// tunnelling across a sub-wavelength gap and the focusing of a lens all come out of it, and the
/// HUD compares them with the textbook formulas. Keys 1-4 choose the scene.
/// Not modelled: the other polarisation (no Brewster angle), the third dimension (the lens is a
/// cylinder), dispersion of the glass, absorption and coatings, and a beam wider than 3
/// wavelengths (it spreads by about 2 degrees, so some light leaks at 45 degrees).
pub struct Light {
    bench: Bench,
    /// fs per second of real time.
    speed: f64,
    texture: OnceCell<Texture2D>,
}

impl Light {
    pub fn new() -> Self {
        Self::with_scene(Scene::Refraction)
    }

    fn with_scene(scene: Scene) -> Self {
        Self { bench: Bench::new(scene, scene.start_angle(), 0.2, Signal::Continuous), speed: START_SPEED, texture: OnceCell::new() }
    }

    /// Changes the angle or the gap and starts again.
    fn rebuild(&mut self, angle: f64, gap: f64) {
        self.bench = Bench::new(self.bench.scene, angle, gap, Signal::Continuous);
    }

    /// Where the lens focuses: the maximum of the mean Ez^2 on the axis, more than 5 cells behind
    /// the flat face, refined by a parabola. Returns the distance behind the flat face (um) and
    /// the intensity there.
    fn lens_focus(report: &Report) -> Option<(f64, f64)> {
        let first = PML + ((LENS_FLAT_FACE / CELL).ceil() as usize) + 5;
        let intensity = &report.axis_intensity;
        let peak = (first..NX - PML - 1).max_by(|&a, &b| intensity[a].partial_cmp(&intensity[b]).unwrap())?;
        let (a, b, c) = (intensity[peak - 1], intensity[peak], intensity[peak + 1]);
        let offset = if a - 2.0 * b + c != 0.0 { 0.5 * (a - c) / (a - 2.0 * b + c) } else { 0.0 };
        let x = (peak as f64 + offset - PML as f64 + 0.5) * CELL;
        Some((x - LENS_FLAT_FACE, b))
    }
}

impl Simulation for Light {
    fn step(&mut self, _dt: f64) {
        self.bench.step();
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
        Rect::new(0.0, 0.0, (INTERIOR_X as f64 * CELL) as f32, (INTERIOR_Y as f64 * CELL) as f32)
    }

    fn handle_input(&mut self) {
        for (key, scene) in [(KeyCode::Key1, Scene::Refraction), (KeyCode::Key2, Scene::TotalReflection), (KeyCode::Key3, Scene::Tunnelling), (KeyCode::Key4, Scene::Lens)] {
            if is_key_pressed(key) {
                let speed = self.speed;
                *self = Self::with_scene(scene);
                self.speed = speed;
            }
        }
        let (angle, gap) = (self.bench.angle, self.bench.gap);
        let direction = is_key_pressed(KeyCode::Right) as i32 - is_key_pressed(KeyCode::Left) as i32;
        if direction != 0 {
            match self.bench.scene {
                Scene::Refraction | Scene::TotalReflection => self.rebuild((angle + 5.0 * direction as f64).clamp(0.0, 60.0), gap),
                Scene::Tunnelling => self.rebuild(angle, (gap + 0.05 * direction as f64).clamp(0.0, 1.0)),
                Scene::Lens => {}
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
        // One pixel per cell of the window: red for positive Ez, blue for negative, and a faint
        // tint on the glass.
        let grid = &self.bench.grid;
        let mut pixels = Vec::with_capacity(4 * INTERIOR_X * INTERIOR_Y);
        let glass_scale = 1.0 / (N_BK7 * N_BK7 - 1.0);
        for j in PML..NY - PML {
            for i in PML..NX - PML {
                let v = grid.ez[j * NX + i].clamp(-1.0, 1.0);
                let tint = ((S / grid.ce[j * NX + i] as f64 - 1.0) * glass_scale) as f32;
                let (r, g, b) = if v > 0.0 { (v, 0.25 * v, 0.0) } else { (0.0, -0.35 * v, -v) };
                let channel = |value: f32, glass: f32| ((value + glass * tint).clamp(0.0, 1.0) * 255.0) as u8;
                pixels.extend_from_slice(&[channel(r, 0.10), channel(g, 0.13), channel(b, 0.16), 255]);
            }
        }
        let texture = self.texture.get_or_init(|| Texture2D::from_rgba8(INTERIOR_X as u16, INTERIOR_Y as u16, &pixels));
        texture.update_from_bytes(INTERIOR_X as u32, INTERIOR_Y as u32, &pixels);
        let size = vec2((INTERIOR_X as f64 * CELL) as f32, (INTERIOR_Y as f64 * CELL) as f32);
        draw_texture_ex(texture, 0.0, 0.0, WHITE, DrawTextureParams { dest_size: Some(size), ..Default::default() });

        // Glass surfaces, and the reflected and refracted rays from the pivot by the law of
        // reflection and Snell's law.
        let width = 0.03;
        let (angle, gap) = (self.bench.angle.to_radians(), self.bench.gap);
        let pivot = vec2(PIVOT.0 as f32, PIVOT.1 as f32);
        let normal = vec2(angle.cos() as f32, angle.sin() as f32);
        let along = vec2(-normal.y, normal.x);
        let line = |from: Vec2, to: Vec2, color: Color| draw_line(from.x, from.y, to.x, to.y, width, color);
        match self.bench.scene {
            Scene::Refraction | Scene::TotalReflection | Scene::Tunnelling => {
                line(pivot - along * 12.0, pivot + along * 12.0, WHITE);
                if self.bench.scene == Scene::Tunnelling {
                    line(pivot + normal * gap as f32 - along * 12.0, pivot + normal * gap as f32 + along * 12.0, WHITE);
                }
                // Reflected ray: d - 2 (d . n) n, for d = +x.
                let reflected = vec2(1.0, 0.0) - 2.0 * normal.x * normal;
                line(pivot, pivot + reflected * 14.0, Color::new(1.0, 1.0, 0.3, 0.8));
                let (n1, n2) = if self.bench.scene == Scene::Refraction { (1.0, N_BK7) } else { (N_BK7, 1.0) };
                let sin_r = n1 * angle.sin() / n2;
                if self.bench.scene != Scene::Tunnelling && sin_r < 1.0 {
                    let direction_angle = angle - sin_r.asin();
                    let refracted = vec2(direction_angle.cos() as f32, direction_angle.sin() as f32);
                    line(pivot, pivot + refracted * 14.0, Color::new(0.3, 1.0, 0.3, 0.8));
                }
            }
            Scene::Lens => {
                draw_poly_lines(PIVOT.0 as f32, PIVOT.1 as f32, 96, LENS_RADIUS as f32, 0.0, width, WHITE);
                let half_height = (LENS_RADIUS * LENS_RADIUS - (LENS_FLAT_FACE - PIVOT.0).powi(2)).sqrt() as f32;
                line(vec2(LENS_FLAT_FACE as f32, PIVOT.1 as f32 - half_height), vec2(LENS_FLAT_FACE as f32, PIVOT.1 as f32 + half_height), WHITE);
            }
        }
    }

    fn hud(&self) -> Vec<String> {
        let bench = &self.bench;
        let angle = bench.angle.to_radians();
        let first = match bench.scene {
            Scene::Refraction => format!("1/4 refraction, vacuum to N-BK7, {:.0} deg (1-4, LEFT/RIGHT)", bench.angle),
            Scene::TotalReflection => format!("2/4 glass to vacuum, {:.0} deg; critical angle 41.3 deg (1-4, LEFT/RIGHT)", bench.angle),
            Scene::Tunnelling => format!("3/4 glass, vacuum gap {:.2} um, glass, {:.0} deg (1-4, LEFT/RIGHT)", bench.gap, bench.angle),
            Scene::Lens => "4/4 plano-convex lens, radius 4 um, N-BK7 (1-4)".to_string(),
        };
        let mut lines = vec![first, format!("t = {:.1} fs   {} fs/s (UP/DOWN)", bench.grid.step_count as f64 * DT, self.speed)];
        let Some(report) = &bench.report else {
            lines.push("measuring...".to_string());
            return lines;
        };
        let (reflectance, transmittance) = (report.reflectance(), report.transmittance());
        let textbook = match bench.scene {
            Scene::Refraction => Some(fresnel_reflectance(1.0, N_BK7, angle)),
            Scene::TotalReflection => Some(fresnel_reflectance(N_BK7, 1.0, angle)),
            Scene::Tunnelling => Some(1.0 - tunnelling_transmittance(angle, bench.gap)),
            Scene::Lens => None,
        };
        if let Some(textbook) = textbook {
            let label = if bench.scene == Scene::Tunnelling { "tunnelling formula" } else { "Fresnel" };
            lines.push(format!(
                "reflected {:.2} % ({} {:.2} %)   transmitted {:.2} %   sum {:.4}",
                100.0 * reflectance,
                label,
                100.0 * textbook,
                100.0 * transmittance,
                reflectance + transmittance
            ));
        }
        match bench.scene {
            Scene::Refraction | Scene::TotalReflection => {
                let (n1, n2) = if bench.scene == Scene::Refraction { (1.0, N_BK7) } else { (N_BK7, 1.0) };
                let snell = n1 * angle.sin() / n2;
                match report.refraction_deg {
                    Some(measured) if transmittance > 0.01 && snell < 1.0 => lines.push(format!("refracted at {:.1} deg (Snell {:.2})", measured, snell.asin().to_degrees())),
                    _ => {
                        let kappa = TAU / WAVELENGTH * (N_BK7 * N_BK7 * angle.sin().powi(2) - 1.0).max(0.0).sqrt();
                        lines.push(format!("critical angle 41.30 deg, evanescent 1/e length {:.2} um", if kappa > 0.0 { 1.0 / kappa } else { f64::INFINITY }));
                    }
                }
            }
            Scene::Lens => {
                if let Some((distance, _)) = Self::lens_focus(report) {
                    lines.push(format!("focus {:.1} um behind the lens (ray optics {:.2})", distance, LENS_RADIUS / (N_BK7 - 1.0) - 1.5 / N_BK7));
                }
            }
            Scene::Tunnelling => {}
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1D grid of `nx` cells with the absorbing layer at both ends, vacuum or all glass.
    fn line(nx: usize, glass: bool) -> (Grid, Beam) {
        let mut grid = Grid::new(nx, 1, PML, 0);
        let n = if glass { N_BK7 } else { 1.0 };
        grid.ce.fill((S / (n * n)) as f32);
        (grid, Beam { column: 0, profile: vec![1.0], index: n, signal: Signal::Continuous })
    }

    /// Runs one period and returns, for each of the `columns`, the flux through that column and
    /// the largest |Ez| there.
    fn one_period(grid: &mut Grid, beam: &Beam, columns: &[usize]) -> Vec<(f64, f64)> {
        let mut result = vec![(0.0, 0.0); columns.len()];
        for _ in 0..STEPS_PER_PERIOD as usize {
            grid.step_h(Some(beam));
            for (r, &c) in result.iter_mut().zip(columns) {
                r.0 += grid.flux_x(c, 0);
            }
            grid.step_e(Some(beam));
            for (r, &c) in result.iter_mut().zip(columns) {
                r.0 += grid.flux_x(c, 0);
                r.1 = f64::max(r.1, grid.ez[c].abs() as f64);
            }
        }
        result
    }

    #[test]
    fn n_bk7_follows_the_sellmeier_formula() {
        let l2 = WAVELENGTH * WAVELENGTH;
        let n2 = 1.0
            + 1.03961212 * l2 / (l2 - 0.00600069867)
            + 0.231792344 * l2 / (l2 - 0.0200179144)
            + 1.01046945 * l2 / (l2 - 103.560653);
        assert!((n2.sqrt() - N_BK7).abs() < 1e-5, "n = {}", n2.sqrt());
        assert!((S - 0.64).abs() < 1e-12);
    }

    #[test]
    fn closed_box_conserves_energy() {
        let mut grid = Grid::new(200, 150, 0, 0);
        for j in 0..150 {
            for i in 0..200 {
                let r2 = ((i as f64 - 120.0).powi(2) + (j as f64 - 70.0).powi(2)).sqrt();
                if r2 <= 40.0 {
                    grid.ce[j * 200 + i] = (S / (N_BK7 * N_BK7)) as f32;
                }
                let r2 = (i as f64 - 60.0).powi(2) + (j as f64 - 80.0).powi(2);
                grid.ez[j * 200 + i] = (-r2 / 100.0).exp() as f32;
            }
        }
        // Energy right after step_h: n^2 Ez^2 + Hx Hx_old + Hy Hy_old, with the old values from before it.
        let energy = |grid: &Grid, old_hx: &[f32], old_hy: &[f32]| {
            let mut sum = 0.0;
            for (k, e) in grid.ez.iter().enumerate() {
                sum += S / grid.ce[k] as f64 * (*e as f64).powi(2);
            }
            sum += grid.hx.iter().zip(old_hx).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>();
            sum += grid.hy.iter().zip(old_hy).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>();
            sum / 2.0
        };
        let (mut first, mut worst): (Option<f64>, f64) = (None, 0.0);
        for _ in 0..20_000 {
            let (old_hx, old_hy) = (grid.hx.clone(), grid.hy.clone());
            grid.step_h(None);
            let e = energy(&grid, &old_hx, &old_hy);
            let e0 = *first.get_or_insert(e);
            worst = worst.max((e / e0 - 1.0).abs());
            grid.step_e(None);
        }
        assert!(worst < 1e-6, "energy drifted by {worst}");
    }

    #[test]
    fn source_is_one_way() {
        for glass in [false, true] {
            let (mut grid, mut beam) = line(3000, glass);
            beam.column = 1500;
            for _ in 0..1500 {
                grid.step_h(Some(&beam));
                grid.step_e(Some(&beam));
            }
            let flux = one_period(&mut grid, &beam, &[1400, 1600]);
            let backward = flux[0].0.abs() / flux[1].0;
            assert!(backward < 1e-6, "glass {glass}: backward / forward power {backward}");
            assert!((flux[1].1 - 1.0).abs() < 0.005, "glass {glass}: amplitude {}", flux[1].1);
        }
    }

    #[test]
    fn absorbing_layer_reflects_little() {
        for glass in [false, true] {
            let (mut grid, mut beam) = line(600, glass);
            beam.column = 100;
            for _ in 0..6000 {
                grid.step_h(Some(&beam));
                grid.step_e(Some(&beam));
            }
            let echo = one_period(&mut grid, &beam, &[50])[0].1;
            assert!(echo < 5e-4, "glass {glass}: amplitude {echo} behind the source");
        }
    }

    /// Phase of Ez at a cell over one period: atan2(sum Ez cos(wt), sum Ez sin(wt)).
    fn phases(grid: &mut Grid, beam: &Beam, cells: &[usize]) -> Vec<f64> {
        let mut sums = vec![(0.0, 0.0); cells.len()];
        for _ in 0..STEPS_PER_PERIOD as usize {
            grid.step_h(Some(beam));
            grid.step_e(Some(beam));
            let t = grid.step_count as f64 * DT;
            for (s, &c) in sums.iter_mut().zip(cells) {
                s.0 += grid.ez[c] as f64 * (OMEGA * t).cos();
                s.1 += grid.ez[c] as f64 * (OMEGA * t).sin();
            }
        }
        sums.iter().map(|s| s.0.atan2(s.1)).collect()
    }

    #[test]
    fn wavelength_is_lambda_over_n() {
        for (glass, yee) in [(false, 19.65365), (true, 29.83974)] {
            let (mut grid, mut beam) = line(600, glass);
            beam.column = 100;
            for _ in 0..6000 {
                grid.step_h(Some(&beam));
                grid.step_e(Some(&beam));
            }
            let n = if glass { N_BK7 } else { 1.0 };
            let phase = phases(&mut grid, &beam, &[300, 400]);
            // The phase difference over 100 cells, plus the multiple of 2 pi nearest to 100 cells
            // of wavelength lambda / n.
            let expected = 200.0 * PI * n / CELLS_PER_WAVELENGTH;
            let raw = phase[0] - phase[1];
            let difference = raw + TAU * ((expected - raw) / TAU).round();
            assert!((difference - yee).abs() < 1e-3, "glass {glass}: phase {difference}, Yee {yee}");
            let index = difference * CELLS_PER_WAVELENGTH / (200.0 * PI);
            assert!((index / n - 1.0).abs() < 0.0035, "glass {glass}: grid index {index}");
        }
    }

    #[test]
    fn pulse_travels_at_c_over_n() {
        for (glass, steps, tolerance) in [(false, 7200, 0.004), (true, 10_000, 0.011)] {
            let (mut grid, mut beam) = line(4000, glass);
            beam.column = 100;
            beam.signal = Signal::Pulse { tau: 4.0 * PERIOD, t0: 12.0 * PERIOD };
            let (mut weight, mut moment) = ([0.0; 2], [0.0; 2]);
            for _ in 0..steps {
                grid.step_h(Some(&beam));
                grid.step_e(Some(&beam));
                let t = grid.step_count as f64 * DT;
                for (k, &c) in [600, 3600].iter().enumerate() {
                    let e2 = (grid.ez[c] as f64).powi(2);
                    weight[k] += e2;
                    moment[k] += e2 * t;
                }
            }
            let (arrival_a, arrival_b) = (moment[0] / weight[0], moment[1] / weight[1]);
            let speed = 3000.0 * CELL / (arrival_b - arrival_a);
            let n = if glass { N_BK7 } else { 1.0 };
            assert!((speed / (C / n) - 1.0).abs() < tolerance, "glass {glass}: speed {speed} um/fs, c/n = {}", C / n);
            // The grid's own group speed, from Yee's dispersion relation.
            let q = (n * (PI / STEPS_PER_PERIOD).sin() / S).asin();
            let yee = C / n * q.cos() / (PI / STEPS_PER_PERIOD).cos();
            assert!((speed / yee - 1.0).abs() < 5e-4, "glass {glass}: speed {speed}, Yee {yee}");
        }
    }

    #[test]
    fn reflectance_at_normal_incidence() {
        let (mut grid, mut beam) = line(800, false);
        beam.column = 100;
        for i in 400..800 {
            grid.ce[i] = (S / (N_BK7 * N_BK7)) as f32;
        }
        for _ in 0..8000 {
            grid.step_h(Some(&beam));
            grid.step_e(Some(&beam));
        }
        let flux = one_period(&mut grid, &beam, &[60, 140, 600]);
        let power = flux[1].0 - flux[0].0;
        let (r, t) = (-flux[0].0 / power, flux[2].0 / power);
        assert!((r - 0.04194).abs() < 0.0015, "R = {r}");
        assert!((r + t - 1.0).abs() < 1e-4, "R + T = {}", r + t);
    }

    #[test]
    fn refraction_at_45_degrees() {
        let mut bench = Bench::new(Scene::Refraction, 45.0, 0.2, Signal::Continuous);
        for _ in 0..3400 {
            bench.step();
        }
        let report = bench.measure(100);
        let (r, t) = (report.reflectance(), report.transmittance());
        assert!((r - fresnel_reflectance(1.0, N_BK7, 45f64.to_radians())).abs() < 0.003, "R = {r}");
        assert!((r + t - 1.0).abs() < 1e-3, "R + T = {}", r + t);
        let snell = (45f64.to_radians().sin() / N_BK7).asin().to_degrees();
        let measured = report.refraction_deg.unwrap();
        assert!((measured - snell).abs() < 0.4, "refraction angle {measured}, Snell {snell}");
    }

    #[test]
    fn absorbing_layer_leaves_no_echo() {
        let mut bench = Bench::new(Scene::Refraction, 60.0, 0.2, Signal::Pulse { tau: 4.0 * PERIOD, t0: 12.0 * PERIOD });
        let energy = |bench: &Bench| {
            let grid = &bench.grid;
            let mut sum = 0.0;
            for j in PML..NY - PML {
                for i in PML..NX - PML {
                    sum += grid.n2(i, j) * (grid.ez[j * NX + i] as f64).powi(2) + (grid.hx[j * NX + i] as f64).powi(2) + (grid.hy[j * (NX + 1) + i] as f64).powi(2);
                }
            }
            sum / 2.0
        };
        let (mut peak, mut last): (f64, f64) = (0.0, 0.0);
        for step in 1..=3500 {
            bench.step();
            if step % 100 == 0 {
                last = energy(&bench);
                peak = peak.max(last);
            }
        }
        assert!(last / peak < 1e-6, "{} of the peak energy is left in the window", last / peak);
    }

    #[test]
    fn total_internal_reflection() {
        // 50 degrees: everything is reflected, and the field outside decays as exp(-kappa d).
        let mut bench = Bench::new(Scene::TotalReflection, 50.0, 0.2, Signal::Continuous);
        for _ in 0..3400 {
            bench.step();
        }
        let r = bench.measure(100).reflectance();
        assert!(r > 0.999, "R = {r} at 50 degrees");

        let angle = 50f64.to_radians();
        let distances: Vec<f64> = (2..=15).map(|k| 0.02 * k as f64).collect();
        let mut intensity = vec![0.0; distances.len()];
        for _ in 0..STEPS_PER_PERIOD as usize {
            bench.step();
            for (sum, d) in intensity.iter_mut().zip(&distances) {
                *sum += bench.grid.ez_at(PIVOT.0 + d * angle.cos(), PIVOT.1 + d * angle.sin()).powi(2);
            }
        }
        // Least-squares slope of ln(intensity) against distance; Ez^2 decays as exp(-2 kappa d).
        let n = distances.len() as f64;
        let ys: Vec<f64> = intensity.iter().map(|v| (v / STEPS_PER_PERIOD).ln()).collect();
        let (mean_d, mean_y) = (distances.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let slope = distances.iter().zip(&ys).map(|(d, y)| (d - mean_d) * (y - mean_y)).sum::<f64>()
            / distances.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>();
        let kappa = -slope / 2.0;
        assert!(kappa > 5.3 && kappa < 6.0, "decay rate {kappa} /um (textbook 5.849)");

        // 45 degrees is only 3.7 degrees above the critical angle, and a narrow beam leaks a little.
        let mut bench = Bench::new(Scene::TotalReflection, 45.0, 0.2, Signal::Continuous);
        for _ in 0..3400 {
            bench.step();
        }
        let r = bench.measure(100).reflectance();
        assert!(r > 0.985 && r < 0.997, "R = {r} at 45 degrees");
    }

    #[test]
    fn light_tunnels_across_a_gap() {
        let mut bench = Bench::new(Scene::Tunnelling, 45.0, 0.2, Signal::Continuous);
        for _ in 0..3400 {
            bench.step();
        }
        let t = bench.measure(100).transmittance();
        let formula = tunnelling_transmittance(45f64.to_radians(), 0.2);
        assert!((formula - 0.3644).abs() < 1e-3, "formula {formula}");
        assert!((t - formula).abs() < 0.02, "T = {t}, formula {formula}");
    }

    #[test]
    fn lens_focuses_short_of_ray_optics() {
        let mut bench = Bench::new(Scene::Lens, 45.0, 0.2, Signal::Continuous);
        for _ in 0..3000 {
            bench.step();
        }
        let report = bench.measure(STEPS_PER_PERIOD as usize);
        let (distance, intensity) = Light::lens_focus(&report).unwrap();
        // Ray optics says 6.78 um from the flat face; spherical aberration and diffraction pull
        // the focus in to about 5.2 um.
        assert!(distance > 4.8 && distance < 5.8, "focus {distance} um behind the flat face");
        assert!(intensity / 0.5 > 2.4, "intensity at the focus is {} times the incident beam's", intensity / 0.5);
    }
}
