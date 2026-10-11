use std::collections::VecDeque;

use macroquad::color::hsl_to_rgb;
use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: SI (metres, kilograms, seconds, radians).
/// Standard gravity, m/s^2.
const G: f64 = 9.80665;

/// The two rods and bobs: masses in kg, rod lengths in m.
struct Params {
    m1: f64,
    m2: f64,
    l1: f64,
    l2: f64,
}

const PENDULUM: Params = Params { m1: 1.0, m2: 1.0, l1: 1.0, l2: 1.0 };

/// `[theta1, theta2, omega1, omega2]`: the angles of the upper and lower rod from the downward
/// vertical (rad, counter-clockwise positive) and their rates (rad/s).
type State = [f64; 4];

/// Angle between neighbouring copies of the pendulum, rad.
const OFFSET: f64 = 1e-6;
const COPIES: usize = 8;

/// Time step in seconds: 8 steps per frame at 60 fps. At this step the error of the integrator is
/// below the round-off of `f64`, so a smaller one would not make the motion any more reliable.
const DT: f64 = 1.0 / 480.0;

// Gauss-Legendre integrator (see `gauss_step`).
const SQRT_15: f64 = 3.872983346207417; // f64::sqrt is not const
const A: [[f64; 3]; 3] = [
    [5.0 / 36.0, 2.0 / 9.0 - SQRT_15 / 15.0, 5.0 / 36.0 - SQRT_15 / 30.0],
    [5.0 / 36.0 + SQRT_15 / 24.0, 2.0 / 9.0, 5.0 / 36.0 - SQRT_15 / 24.0],
    [5.0 / 36.0 + SQRT_15 / 30.0, 2.0 / 9.0 + SQRT_15 / 15.0, 5.0 / 36.0],
];
const B: [f64; 3] = [5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0];
/// The fixed-point iteration stops when no component of the stage derivatives changes by more than
/// this (relative to their size)...
const TOL: f64 = 1e-13;
/// ...or after this many sweeps.
const MAX_SWEEPS: usize = 50;

/// Distance between the reference and the shadow trajectory of the Lyapunov estimate.
const SHADOW_DISTANCE: f64 = 1e-8;

/// Energy scale `(M l1 + m2 l2) g`, which is also minus the energy hanging straight down (the
/// lowest the pendulum can have). Drift is measured against it, since the energy itself can be 0.
const E_SCALE: f64 = ((PENDULUM.m1 + PENDULUM.m2) * PENDULUM.l1 + PENDULUM.m2 * PENDULUM.l2) * G;
const E_REST: f64 = -E_SCALE;

// Drawing, in metres with the pivot at the origin.
const ROD_WIDTH: f32 = 0.04;
const BOB_RADIUS: f32 = 0.07;
const CURVE_WIDTH: f32 = 0.02;
/// One trail point every this many steps, and the trail keeps this many (1 s).
const TRAIL_EVERY: u64 = 4;
const TRAIL_LEN: usize = 120;
/// Plot of the separation of each copy from the reference: one row every this many steps (0.1 s),
/// for up to this many rows (40 s).
const PLOT_EVERY: u64 = 48;
const PLOT_ROWS: usize = 401;
const PLOT_SECONDS_PER_ROW: f32 = 0.1;
/// The plot's rectangle: x from `PLOT_LEFT` to `PLOT_RIGHT` for 40 s, y from `PLOT_BOTTOM` to
/// `PLOT_TOP` for separations of 10^LOG_MIN to 10^LOG_MAX rad.
const PLOT_LEFT: f32 = 2.9;
const PLOT_RIGHT: f32 = 7.6;
const PLOT_BOTTOM: f32 = -2.0;
const PLOT_TOP: f32 = 2.0;
const LOG_MIN: f32 = -7.0;
const LOG_MAX: f32 = 1.0;

const START_RELEASE_DEG: f64 = 120.0;
const MIN_RELEASE_DEG: f64 = 10.0;
const MAX_RELEASE_DEG: f64 = 170.0;

/// Time derivative of the state. These solve Lagrange's equations for the two angles,
///   M l1 theta1'' + m2 l2 cos(d) theta2'' = r1,   l1 cos(d) theta1'' + l2 theta2'' = r2,
/// with d = theta1 - theta2 and M = m1 + m2.
fn derivative(p: &Params, y: State) -> State {
    let [theta1, theta2, omega1, omega2] = y;
    let (s, c) = (theta1 - theta2).sin_cos();
    let total_m = p.m1 + p.m2;
    let r1 = -p.m2 * p.l2 * omega2 * omega2 * s - total_m * G * theta1.sin();
    let r2 = p.l1 * omega1 * omega1 * s - G * theta2.sin();
    let det = p.m1 + p.m2 * s * s; // always positive
    [
        omega1,
        omega2,
        (r1 - p.m2 * c * r2) / (p.l1 * det),
        (total_m * r2 - c * r1) / (p.l2 * det),
    ]
}

/// Kinetic plus potential energy in joules, with the potential zero at the pivot's height.
fn energy(p: &Params, y: State) -> f64 {
    let [theta1, theta2, omega1, omega2] = y;
    let total_m = p.m1 + p.m2;
    let kinetic = 0.5 * total_m * p.l1 * p.l1 * omega1 * omega1
        + 0.5 * p.m2 * p.l2 * p.l2 * omega2 * omega2
        + p.m2 * p.l1 * p.l2 * omega1 * omega2 * (theta1 - theta2).cos();
    let potential = -total_m * G * p.l1 * theta1.cos() - p.m2 * G * p.l2 * theta2.cos();
    kinetic + potential
}

/// One step of the 3-stage Gauss-Legendre method (implicit Runge-Kutta, sixth order).
/// It is time-symmetric and symplectic, so the energy error oscillates instead of growing. That
/// matters here: the kinetic energy depends on the angles, which rules out leapfrog.
///
/// The stage derivatives satisfy `K_i = f(y + h sum_j A_ij K_j)`, found by fixed-point iteration.
fn gauss_step(p: &Params, y: State, h: f64) -> State {
    let mut k = [derivative(p, y); 3];
    for _ in 0..MAX_SWEEPS {
        let mut new_k = [[0.0; 4]; 3];
        for i in 0..3 {
            let mut arg = y;
            for j in 0..3 {
                for c in 0..4 {
                    arg[c] += h * A[i][j] * k[j][c];
                }
            }
            new_k[i] = derivative(p, arg);
        }

        let mut change: f64 = 0.0;
        let mut size: f64 = 0.0;
        for i in 0..3 {
            for c in 0..4 {
                change = change.max((new_k[i][c] - k[i][c]).abs());
                size = size.max(new_k[i][c].abs());
            }
        }
        k = new_k;
        if change <= TOL * (1.0 + size) {
            break;
        }
    }

    let mut next = y;
    for c in 0..4 {
        next[c] += h * (B[0] * k[0][c] + B[1] * k[1][c] + B[2] * k[2][c]);
    }
    next
}

/// Maps an angle into [-pi, pi). Without it the angles grow without bound and cost accuracy.
fn wrap_angle(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// A `gauss_step` followed by wrapping both angles.
fn advance(p: &Params, y: State, h: f64) -> State {
    let mut next = gauss_step(p, y, h);
    next[0] = wrap_angle(next[0]);
    next[1] = wrap_angle(next[1]);
    next
}

/// `a - b`, with the two angle differences wrapped (angles that straddle +-pi are close).
fn difference(a: State, b: State) -> State {
    [wrap_angle(a[0] - b[0]), wrap_angle(a[1] - b[1]), a[2] - b[2], a[3] - b[3]]
}

/// Distance between two states in radians. The rates are multiplied by sqrt(l1/g) (0.32 s) to
/// put all four terms in the same unit.
fn separation(a: State, b: State) -> f64 {
    let d = difference(a, b);
    (d[0] * d[0] + d[1] * d[1] + (d[2] * d[2] + d[3] * d[3]) * PENDULUM.l1 / G).sqrt()
}

/// Positions of the upper and lower bob, pivot at the origin, y up.
fn bob_positions(p: &Params, y: State) -> (Vec2, Vec2) {
    let upper = dvec2(p.l1 * y[0].sin(), -p.l1 * y[0].cos());
    let lower = upper + dvec2(p.l2 * y[1].sin(), -p.l2 * y[1].cos());
    (upper.as_vec2(), lower.as_vec2())
}

/// Running estimate of the largest Lyapunov exponent: the average rate at which two neighbouring
/// trajectories separate, `d(t) ~ d(0) e^(lambda t)`. Benettin's method: a shadow trajectory runs
/// next to the reference and is pulled back to a fixed small distance after every step, adding up
/// the logarithm of how much it stretched.
struct LyapunovEstimate {
    shadow: State,
    log_stretch: f64,
    time: f64,
}

impl LyapunovEstimate {
    fn new(reference: State) -> Self {
        let mut shadow = reference;
        shadow[0] += SHADOW_DISTANCE;
        Self { shadow, log_stretch: 0.0, time: 0.0 }
    }

    /// Call after the reference has been advanced by `dt`.
    fn update(&mut self, reference: State, dt: f64) {
        self.shadow = advance(&PENDULUM, self.shadow, dt);
        let diff = difference(self.shadow, reference);
        let distance = separation(self.shadow, reference);
        self.log_stretch += (distance / SHADOW_DISTANCE).ln();
        self.time += dt;
        // Back to the original distance, in the same direction.
        let scale = SHADOW_DISTANCE / distance;
        self.shadow = reference;
        for c in 0..4 {
            self.shadow[c] += diff[c] * scale;
        }
    }

    /// Average growth rate so far, 1/s.
    fn value(&self) -> f64 {
        if self.time > 0.0 { self.log_stretch / self.time } else { 0.0 }
    }
}

/// Eight identical double pendulums (two point masses on massless rigid rods, no friction, in a
/// vertical plane), released from rest as a straight stick, each one a millionth of a radian
/// further than the last. They move as one for several seconds, then fan out: deterministic chaos.
/// A log plot shows each copy's separation from the first, and the HUD the Lyapunov exponent.
/// Not modelled: friction, the mass and size of real rods, collisions between the parts, and
/// individual trajectories beyond ~20 s (any error, even the last bit of an `f64`, grows with the
/// Lyapunov exponent; the energy and the statistics stay right).
pub struct DoublePendulum {
    release_deg: f64,
    /// `copies[0]` is the reference.
    copies: [State; COPIES],
    /// Recent lower-bob positions of each copy.
    trails: [VecDeque<Vec2>; COPIES],
    /// log10 of each copy's (1 to 7) separation from the reference, one row per 0.1 s.
    plot: Vec<[f32; COPIES - 1]>,
    lyapunov: LyapunovEstimate,
    energy0: f64,
    t: f64,
    steps_since_gather: u64,
}

impl DoublePendulum {
    pub fn new() -> Self {
        Self::released_from(START_RELEASE_DEG)
    }

    fn released_from(release_deg: f64) -> Self {
        let theta = release_deg.to_radians();
        let reference = [theta, theta, 0.0, 0.0];
        let mut pendulum = Self {
            release_deg,
            copies: [reference; COPIES],
            trails: Default::default(),
            plot: Vec::new(),
            lyapunov: LyapunovEstimate::new(reference),
            energy0: energy(&PENDULUM, reference),
            t: 0.0,
            steps_since_gather: 0,
        };
        pendulum.gather();
        pendulum
    }

    /// Re-forms the bundle around the reference: copy `k` gets both angles `k * OFFSET` larger.
    fn gather(&mut self) {
        let reference = self.copies[0];
        for k in 1..COPIES {
            self.copies[k] = reference;
            self.copies[k][0] += k as f64 * OFFSET;
            self.copies[k][1] += k as f64 * OFFSET;
            self.trails[k].clear();
        }
        self.plot.clear();
        self.push_plot_row();
        self.steps_since_gather = 0;
    }

    fn push_plot_row(&mut self) {
        let mut row = [0.0; COPIES - 1];
        for k in 1..COPIES {
            row[k - 1] = separation(self.copies[k], self.copies[0]).log10() as f32;
        }
        self.plot.push(row);
    }

    /// The largest separation of any copy from the reference.
    fn spread(&self) -> f64 {
        (1..COPIES).map(|k| separation(self.copies[k], self.copies[0])).fold(0.0, f64::max)
    }

    fn color(k: usize) -> Color {
        if k == 0 { WHITE } else { hsl_to_rgb(k as f32 / COPIES as f32, 1.0, 0.6) }
    }

    /// Plot coordinates (world) of a time in seconds and a log10 separation.
    fn plot_point(seconds: f32, log_separation: f32) -> Vec2 {
        let x = PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT) * seconds / ((PLOT_ROWS - 1) as f32 * PLOT_SECONDS_PER_ROW);
        let y = PLOT_BOTTOM + (PLOT_TOP - PLOT_BOTTOM) * (log_separation - LOG_MIN) / (LOG_MAX - LOG_MIN);
        vec2(x, y)
    }

    fn draw_plot(&self) {
        let grid = Color::new(1.0, 1.0, 1.0, 0.15);
        let top_seconds = (PLOT_ROWS - 1) as f32 * PLOT_SECONDS_PER_ROW;
        // A horizontal line per decade, a vertical line every 10 s.
        for decade in (LOG_MIN as i32)..=(LOG_MAX as i32) {
            let (a, b) = (Self::plot_point(0.0, decade as f32), Self::plot_point(top_seconds, decade as f32));
            draw_line(a.x, a.y, b.x, b.y, 0.01, grid);
        }
        for tick in 0..=4 {
            let seconds = 10.0 * tick as f32;
            let (a, b) = (Self::plot_point(seconds, LOG_MIN), Self::plot_point(seconds, LOG_MAX));
            draw_line(a.x, a.y, b.x, b.y, 0.01, grid);
        }

        // What steady growth at the average rate would look like, from where copy 1 starts.
        let lambda = self.lyapunov.value();
        if lambda > 0.1 {
            let start = (2f64.sqrt() * OFFSET).log10() as f32;
            let per_second = (lambda / std::f64::consts::LN_10) as f32;
            let end = ((LOG_MAX - start) / per_second).min(top_seconds); // where it leaves the plot
            let (a, b) = (Self::plot_point(0.0, start), Self::plot_point(end, start + per_second * end));
            draw_line(a.x, a.y, b.x, b.y, CURVE_WIDTH, GRAY);
        }

        for k in 1..COPIES {
            let points = self
                .plot
                .iter()
                .enumerate()
                .map(|(row, values)| Self::plot_point(row as f32 * PLOT_SECONDS_PER_ROW, values[k - 1].clamp(LOG_MIN, LOG_MAX)));
            for (p0, p1) in points.clone().zip(points.skip(1)) {
                draw_line(p0.x, p0.y, p1.x, p1.y, CURVE_WIDTH, Self::color(k));
            }
        }
    }
}

impl Simulation for DoublePendulum {
    fn step(&mut self, dt: f64) {
        for copy in &mut self.copies {
            *copy = advance(&PENDULUM, *copy, dt);
        }
        self.lyapunov.update(self.copies[0], dt);
        self.t += dt;
        self.steps_since_gather += 1;

        if self.steps_since_gather % TRAIL_EVERY == 0 {
            for (copy, trail) in self.copies.iter().zip(&mut self.trails) {
                if trail.len() == TRAIL_LEN {
                    trail.pop_front();
                }
                trail.push_back(bob_positions(&PENDULUM, *copy).1);
            }
        }
        if self.steps_since_gather % PLOT_EVERY == 0 && self.plot.len() < PLOT_ROWS {
            self.push_plot_row();
        }
    }

    fn is_over(&self) -> bool {
        false // runs indefinitely
    }

    fn world_bounds(&self) -> Rect {
        // Aspect 1.5 like the default window; the pendulums reach a circle of radius 2 around the
        // pivot, the plot is to the right, and the band above y = 2 is left for the HUD.
        Rect::new(-2.3, -2.3, 10.2, 6.8)
    }

    fn draw(&self) {
        self.draw_plot();

        // Trails, fading from transparent to the copy's color.
        for (k, trail) in self.trails.iter().enumerate() {
            let n = trail.len();
            let color = Self::color(k);
            for (i, (p0, p1)) in trail.iter().zip(trail.iter().skip(1)).enumerate() {
                let alpha = (i as f32 + 1.0) / n as f32;
                draw_line(p0.x, p0.y, p1.x, p1.y, CURVE_WIDTH, Color::new(color.r, color.g, color.b, alpha * 0.8));
            }
        }

        // Rods and bobs, the reference (copy 0) last so it is on top.
        for k in (0..COPIES).rev() {
            let (upper, lower) = bob_positions(&PENDULUM, self.copies[k]);
            let color = Self::color(k);
            draw_line(0.0, 0.0, upper.x, upper.y, ROD_WIDTH, color);
            draw_line(upper.x, upper.y, lower.x, lower.y, ROD_WIDTH, color);
            draw_circle(upper.x, upper.y, BOB_RADIUS, color);
            draw_circle(lower.x, lower.y, BOB_RADIUS, color);
        }
        draw_circle(0.0, 0.0, 0.03, GRAY);
    }

    fn dt(&self) -> f64 {
        DT
    }

    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Up) {
            *self = Self::released_from((self.release_deg + 10.0).min(MAX_RELEASE_DEG));
        }
        if is_key_pressed(KeyCode::Down) {
            *self = Self::released_from((self.release_deg - 10.0).max(MIN_RELEASE_DEG));
        }
        if is_key_pressed(KeyCode::R) {
            *self = Self::released_from(self.release_deg);
        }
        if is_key_pressed(KeyCode::B) {
            self.gather();
        }
    }

    fn hud(&self) -> Vec<String> {
        let lambda = self.lyapunov.value();
        let mut lyapunov = format!("Lyapunov = {:.2} /s over {:.0} s", lambda, self.lyapunov.time);
        if lambda > 0.1 {
            lyapunov += &format!(" (x10 every {:.1} s)", std::f64::consts::LN_10 / lambda);
        }
        let e = energy(&PENDULUM, self.copies[0]);
        vec![
            format!("t = {:.1} s   release {:.0} deg (UP/DOWN)", self.t, self.release_deg),
            format!("energy {:.2} J above rest   drift {:+.1e}", e - E_REST, (e - self.energy0) / E_SCALE),
            format!("spread {:.1e} rad   R: restart   B: re-bundle", self.spread()),
            lyapunov,
            "plot: separation, 1e-7..10 rad (log) over 40 s".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state released straight, both rods at `deg` from the vertical, at rest.
    fn straight(deg: f64) -> State {
        let theta = deg.to_radians();
        [theta, theta, 0.0, 0.0]
    }

    /// Runs `steps` wrapped steps of size `h`.
    fn run(p: &Params, mut y: State, h: f64, steps: usize) -> State {
        for _ in 0..steps {
            y = advance(p, y, h);
        }
        y
    }

    /// Average period of `theta1` from `cycles` cycles, timed at upward zero crossings
    /// (interpolated linearly between steps).
    fn period(p: &Params, mut y: State, cycles: usize) -> f64 {
        let (mut crossings, mut first, mut last) = (0, 0.0, 0.0);
        let mut t = 0.0;
        while crossings <= cycles {
            let previous = y[0];
            y = advance(p, y, DT);
            t += DT;
            if previous < 0.0 && y[0] >= 0.0 {
                let crossing = t - DT * y[0] / (y[0] - previous);
                if crossings == 0 {
                    first = crossing;
                }
                last = crossing;
                crossings += 1;
            }
        }
        (last - first) / cycles as f64
    }

    #[test]
    fn accelerations_and_energy_match_reference_values() {
        let y = [1.0, -0.5, 0.7, -1.2];
        let unit = Params { m1: 1.0, m2: 1.0, l1: 1.0, l2: 1.0 };
        let a = derivative(&unit, y);
        assert!((a[2] - -9.176741637194805).abs() < 1e-12, "theta1'' {}", a[2]);
        assert!((a[3] - 5.839468025431549).abs() < 1e-12, "theta2'' {}", a[3]);
        assert!((energy(&unit, y) - -18.05267549564668).abs() < 1e-12, "energy {}", energy(&unit, y));

        let other = Params { m1: 2.0, m2: 0.5, l1: 1.2, l2: 0.7 };
        let a = derivative(&other, y);
        assert!((a[2] - -7.113718470094819).abs() < 1e-12, "theta1'' {}", a[2]);
        assert!((a[3] - 8.417044222725320).abs() < 1e-12, "theta2'' {}", a[3]);
        assert!((energy(&other, y) - -17.87437366897052).abs() < 1e-12, "energy {}", energy(&other, y));
    }

    #[test]
    fn small_swings_have_the_normal_mode_periods() {
        // Two normal modes with theta2 = +-sqrt(2) theta1 and omega^2 = (2 -+ sqrt(2)) g/l.
        let root2 = 2f64.sqrt();
        let slow = period(&PENDULUM, [1e-3, root2 * 1e-3, 0.0, 0.0], 10);
        let fast = period(&PENDULUM, [1e-3, -root2 * 1e-3, 0.0, 0.0], 10);
        assert!((slow / 2.62150007408 - 1.0).abs() < 1e-6, "slow period {slow}");
        assert!((fast / 1.08586088445 - 1.0).abs() < 1e-5, "fast period {fast}");
    }

    #[test]
    fn without_a_lower_mass_it_is_a_simple_pendulum() {
        // With m2 = 0, theta1'' = -g/l sin(theta1). Released at 90 degrees the period is
        // 4 sqrt(l/g) K(sin 45 deg), K the complete elliptic integral (1.854074677301372).
        let p = Params { m1: 1.0, m2: 0.0, l1: 1.0, l2: 1.0 };
        let expected = 4.0 * (1.0 / G).sqrt() * 1.854074677301372;
        let measured = period(&p, [std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2, 0.0, 0.0], 5);
        assert!((measured / expected - 1.0).abs() < 1e-8, "period {measured}, expected {expected}");
    }

    #[test]
    fn conserves_energy() {
        for deg in [120.0, 170.0] {
            let mut y = straight(deg);
            let e0 = energy(&PENDULUM, y);
            let mut worst: f64 = 0.0;
            for _ in 0..(600.0 / DT) as usize {
                y = advance(&PENDULUM, y, DT);
                worst = worst.max((energy(&PENDULUM, y) - e0).abs() / E_SCALE);
            }
            assert!(worst < 1e-10, "energy drift {worst} at {deg} degrees");
        }
    }

    #[test]
    fn running_backwards_returns_to_the_start() {
        // The method is time-symmetric. The chaotic case is only good for a few seconds because
        // round-off grows in both directions.
        for (deg, steps, limit) in [(120.0, 2400, 1e-10), (30.0, 9600, 1e-12)] {
            let start = straight(deg);
            let there = run(&PENDULUM, start, DT, steps);
            let back = run(&PENDULUM, there, -DT, steps);
            let error = separation(back, start);
            assert!(error < limit, "{deg} degrees: off by {error} after going back");
        }
    }

    #[test]
    fn step_sizes_agree_while_predictable() {
        let start = straight(120.0);
        let coarse = run(&PENDULUM, start, DT, 4800); // 10 s
        let fine = run(&PENDULUM, start, DT / 2.0, 9600);
        let error = separation(coarse, fine);
        assert!(error < 1e-8, "steps differ by {error} after 10 s");
    }

    #[test]
    fn integrator_is_sixth_order() {
        // Halving the step should shrink the error 2^6 = 64 times. The reference is a much finer
        // run. Do not use 1/30 s here: that step is too coarse to be in the asymptotic regime.
        let start = straight(120.0);
        let reference = run(&PENDULUM, start, 1.0 / 960.0, 1920);
        let error = |h: f64| {
            let y = run(&PENDULUM, start, h, (2.0 / h).round() as usize);
            (0..4).map(|c| (y[c] - reference[c]).abs()).fold(0.0, f64::max)
        };
        let ratio = error(1.0 / 60.0) / error(1.0 / 120.0);
        assert!(ratio > 50.0 && ratio < 80.0, "error ratio {ratio}");
    }

    #[test]
    fn chaotic_bundle_stays_together_then_fans_out() {
        let (mut a, mut b) = (straight(120.0), straight(120.0));
        b[0] += OFFSET;
        b[1] += OFFSET;
        let mut fanned_out_at = None;
        for step in 1..=(15.0 / DT) as usize {
            a = advance(&PENDULUM, a, DT);
            b = advance(&PENDULUM, b, DT);
            let d = separation(a, b);
            let t = step as f64 * DT;
            if t <= 4.0 {
                assert!(d < 1e-4, "separated by {d} already at {t} s");
            }
            if d > 1.0 && fanned_out_at.is_none() {
                fanned_out_at = Some(t);
            }
        }
        assert!(fanned_out_at.is_some(), "still together after 15 s");
    }

    #[test]
    fn regular_bundle_stays_together() {
        let (mut a, mut b) = (straight(30.0), straight(30.0));
        b[0] += OFFSET;
        b[1] += OFFSET;
        let mut worst: f64 = 0.0;
        for _ in 0..(60.0 / DT) as usize {
            a = advance(&PENDULUM, a, DT);
            b = advance(&PENDULUM, b, DT);
            worst = worst.max(separation(a, b));
        }
        assert!(worst < 5e-5, "separated by {worst}");
    }

    #[test]
    fn lyapunov_exponent_tells_chaos_from_regular_motion() {
        let estimate = |deg: f64| {
            let mut y = straight(deg);
            let mut lyapunov = LyapunovEstimate::new(y);
            for _ in 0..(300.0 / DT) as usize {
                y = advance(&PENDULUM, y, DT);
                lyapunov.update(y, DT);
            }
            lyapunov.value()
        };
        let chaotic = estimate(120.0);
        let regular = estimate(30.0);
        assert!(chaotic > 1.1 && chaotic < 1.8, "chaotic exponent {chaotic}");
        assert!(regular < 0.05, "regular exponent {regular}");
    }
}
