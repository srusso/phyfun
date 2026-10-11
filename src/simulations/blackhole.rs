use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: geometric, G = c = 1 and the black hole's mass M = 1. Lengths are in GM/c^2, times in
// GM/c^3 and speeds in c. `M` never appears in the code; "in units of M" means these.

// Sagittarius A*, to put real numbers on the HUD. Mass from GRAVITY Collaboration, A&A 2022
// (arXiv:2112.07478): 4.297e6 +- 0.012e6 (statistical) +- 0.040e6 (systematic) solar masses.
const SGR_A_SOLAR_MASSES: f64 = 4.297e6;
const GM_SUN: f64 = 132_712_440_041.939_38; // km^3/s^2
const C: f64 = 299_792.458; // km/s
/// Seconds of far-away time in one unit of M for Sagittarius A*: GM/c^3.
const SGR_A_SECONDS_PER_M: f64 = GM_SUN / (C * C * C) * SGR_A_SOLAR_MASSES;

/// Time step in M. The photon sphere's period is 32.6 M (650 steps), the ISCO's 92 M. RK4 at
/// this step is already at rounding accuracy for the scenes here.
const DT: f64 = 0.05;

/// A body that gets this close to the horizon (r = 2) is drawn frozen there: seen from far away,
/// nothing ever crosses it.
const CAPTURE_RADIUS: f64 = 2.01;
/// Critical impact parameter of light: rays with less are captured.
const CRITICAL_IMPACT: f64 = 5.196_152_422_706_632; // 3 sqrt(3)
/// A trail point is added when a body has moved this far, as a fraction of its radius.
const TRAIL_SPACING: f32 = 0.1;

const RAYS: i32 = 30; // rays at n = -RAYS..=RAYS, 0.4 M apart

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Massive test particle on a Schwarzschild geodesic.
    Particle,
    /// Light.
    Photon,
    /// Newtonian gravity for comparison.
    Newtonian,
}

#[derive(Clone, Copy, PartialEq)]
enum Fate {
    Moving,
    Captured,
    Escaped,
}

#[derive(Clone, Copy, PartialEq)]
enum Scene {
    Orbit,
    Isco,
    Light,
}

/// Position in the orbital plane, `u = d pos/d tau` (proper time for a particle, an affine
/// parameter for a photon) and the elapsed `tau`.
#[derive(Clone, Copy)]
struct State {
    pos: DVec2,
    u: DVec2,
    tau: f64,
}

impl State {
    /// `self + h * k`, for the Runge-Kutta stages.
    fn plus(&self, k: &State, h: f64) -> State {
        State { pos: self.pos + k.pos * h, u: self.u + k.u * h, tau: self.tau + k.tau * h }
    }
}

struct Body {
    kind: Kind,
    state: State,
    /// Conserved energy and angular momentum per unit mass, set once at the start.
    energy: f64,
    angular_momentum: f64,
    fate: Fate,
    color: Color,
    trail: VecDeque<Vec2>,
    trail_len: usize,
    /// The angle phi around the hole, unwrapped (rad).
    turns: f64,
    /// Angle at the last apoapsis (the farthest point of the orbit) and the angle swept since the
    /// one before.
    last_apoapsis: Option<f64>,
    sweep: Option<f64>,
}

impl Body {
    fn new(kind: Kind, pos: DVec2, u: DVec2, energy: f64, color: Color, trail_len: usize) -> Self {
        assert!(pos.length() > 2.0, "start inside the horizon");
        Self {
            kind,
            state: State { pos, u, tau: 0.0 },
            energy,
            angular_momentum: pos.perp_dot(u),
            fate: Fate::Moving,
            color,
            trail: VecDeque::new(),
            trail_len,
            turns: 0.0,
            last_apoapsis: None,
            sweep: None,
        }
    }

    /// 1 for a particle, 0 for a photon (the squared "mass" in the geodesic equations).
    fn kappa(&self) -> f64 {
        if self.kind == Kind::Photon { 0.0 } else { 1.0 }
    }

    /// A massive particle from position `pos` with velocity `u` (per unit proper time).
    fn particle(pos: DVec2, u: DVec2, color: Color, trail_len: usize) -> Self {
        let mut body = Self::new(Kind::Particle, pos, u, 0.0, color, trail_len);
        body.energy = body.energy_now();
        body
    }

    /// A light ray at `pos` moving in the direction `direction` (a unit vector) as drawn.
    fn photon(pos: DVec2, direction: DVec2, color: Color, trail_len: usize) -> Self {
        let r = pos.length();
        let radial = direction.dot(pos) / r;
        // Scales the direction so that the energy is exactly 1.
        let k = 1.0 / (radial * radial + (1.0 - 2.0 / r) * (1.0 - radial * radial)).sqrt();
        Self::new(Kind::Photon, pos, direction * k, 1.0, color, trail_len)
    }

    /// A Newtonian test particle, started like `particle` (same `pos` and `u`).
    fn newtonian(pos: DVec2, v: DVec2, color: Color, trail_len: usize) -> Self {
        Self::new(Kind::Newtonian, pos, v, 0.0, color, trail_len)
    }

    /// Bound orbit with semi-latus rectum `p` and eccentricity `e` (turning points p/(1+e) and
    /// p/(1-e), stable if p > 6 + 2e), started at the farthest point.
    fn bound_orbit(p: f64, e: f64, color: Color, trail_len: usize) -> Self {
        let l = p / (p - 3.0 - e * e).sqrt();
        let r = p / (1.0 - e);
        let mut body = Self::particle(dvec2(r, 0.0), dvec2(0.0, l / r), color, trail_len);
        body.last_apoapsis = Some(0.0);
        body
    }

    /// Circular orbit at radius `r` > 3 (stable beyond 6).
    fn circular_orbit(r: f64, color: Color, trail_len: usize) -> Self {
        Self::particle(dvec2(r, 0.0), dvec2(0.0, 1.0 / (r - 3.0).sqrt()), color, trail_len)
    }

    /// d tau / d t: how fast this body's own clock runs against the far-away one.
    fn rate(&self) -> f64 {
        match self.kind {
            Kind::Newtonian => 1.0,
            _ => (1.0 - 2.0 / self.state.pos.length()) / self.energy,
        }
    }

    /// Time derivative of the state in far-away time `t`, so every body shares one clock.
    ///   pos' = u,   u' = -(kappa / r^3 + 3 L^2 / r^5) pos,   all times d tau/dt.
    /// The first term is Newton's gravity; the L^2 term is all of general relativity here (the
    /// precession, the ISCO, the plunge, the bending of light) and is the only one acting on light.
    fn derivative(&self, s: &State) -> State {
        let r2 = s.pos.length_squared();
        let r = r2.sqrt();
        let (rate, force) = match self.kind {
            Kind::Newtonian => (1.0, 1.0 / (r2 * r)),
            _ => {
                let l2 = self.angular_momentum * self.angular_momentum;
                ((1.0 - 2.0 / r) / self.energy, self.kappa() / (r2 * r) + 3.0 * l2 / (r2 * r2 * r))
            }
        };
        State { pos: s.u * rate, u: s.pos * (-force * rate), tau: rate }
    }

    /// Energy recomputed from the current state: E^2 = u_r^2 + (1 - 2/r)(kappa + L^2/r^2).
    fn energy_now(&self) -> f64 {
        let State { pos, u, .. } = self.state;
        let r = pos.length();
        let u_r = pos.dot(u) / r;
        let l = self.angular_momentum_now();
        (u_r * u_r + (1.0 - 2.0 / r) * (self.kappa() + l * l / (r * r))).sqrt()
    }

    /// Angular momentum recomputed from the current state: L = x u_y - y u_x.
    fn angular_momentum_now(&self) -> f64 {
        self.state.pos.perp_dot(self.state.u)
    }

    /// One fourth-order Runge-Kutta step of `dt` (far-away time), then the bookkeeping.
    fn step(&mut self, dt: f64, escape_radius: f64) {
        if self.fate != Fate::Moving {
            return;
        }
        let before = self.state;
        let k1 = self.derivative(&before);
        let k2 = self.derivative(&before.plus(&k1, dt / 2.0));
        let k3 = self.derivative(&before.plus(&k2, dt / 2.0));
        let k4 = self.derivative(&before.plus(&k3, dt));
        self.state = State {
            pos: before.pos + (k1.pos + (k2.pos + k3.pos) * 2.0 + k4.pos) * (dt / 6.0),
            u: before.u + (k1.u + (k2.u + k3.u) * 2.0 + k4.u) * (dt / 6.0),
            tau: before.tau + (k1.tau + (k2.tau + k3.tau) * 2.0 + k4.tau) * (dt / 6.0),
        };

        // Unwrapped angle: add the change in angle, wrapped to (-pi, pi].
        let turns_before = self.turns;
        let change = self.state.pos.to_angle() - before.pos.to_angle();
        self.turns += (change + PI).rem_euclid(TAU) - PI;

        // Apoapsis: where r r' = pos . u goes from positive to zero or below. Interpolate the angle.
        let (s_before, s_after) = (before.pos.dot(before.u), self.state.pos.dot(self.state.u));
        if s_before > 0.0 && s_after <= 0.0 {
            let angle = turns_before + s_before / (s_before - s_after) * (self.turns - turns_before);
            if let Some(last) = self.last_apoapsis {
                self.sweep = Some(angle - last);
            }
            self.last_apoapsis = Some(angle);
        }

        let r = self.state.pos.length();
        if r < CAPTURE_RADIUS {
            self.fate = Fate::Captured;
        } else if r > escape_radius && s_after > 0.0 {
            self.fate = Fate::Escaped;
        }

        let pos = self.state.pos.as_vec2();
        if self.trail.back().is_none_or(|last| last.distance(pos) >= TRAIL_SPACING * r as f32) {
            if self.trail.len() == self.trail_len {
                self.trail.pop_front();
            }
            self.trail.push_back(pos);
        }
    }
}

/// Complete elliptic integral of the first kind, with parameter `m` (= k^2), by the
/// arithmetic-geometric mean.
fn elliptic_k(m: f64) -> f64 {
    let (mut a, mut b) = (1.0, (1.0 - m).sqrt());
    for _ in 0..40 {
        (a, b) = ((a + b) / 2.0, (a * b).sqrt());
    }
    PI / (2.0 * a)
}

/// Angle swept between two apoapsis passages of a bound orbit with semi-latus rectum `p` and
/// eccentricity `e`: 2 pi in flat space, more here. For large p the excess tends to 6 pi / p.
fn apoapsis_sweep(p: f64, e: f64) -> f64 {
    let d = p - 6.0 + 2.0 * e;
    4.0 * (p / d).sqrt() * elliptic_k(4.0 * e / d)
}

/// A time in M as it would be at Sagittarius A*: minutes below an hour, else hours.
fn sgr_a_time(t: f64) -> String {
    let seconds = t * SGR_A_SECONDS_PER_M;
    if seconds < 3600.0 { format!("{:.1} min", seconds / 60.0) } else { format!("{:.2} h", seconds / 3600.0) }
}

impl Scene {
    fn bounds(self) -> Rect {
        match self {
            Scene::Orbit => Rect::new(-66.0, -44.0, 132.0, 88.0),
            Scene::Isco => Rect::new(-18.0, -12.0, 36.0, 24.0),
            Scene::Light => Rect::new(-30.0, -20.0, 60.0, 40.0),
        }
    }

    /// Simulated M per second of real time at the start.
    fn start_speed(self) -> f64 {
        match self {
            Scene::Orbit => 100.0,
            Scene::Isco => 20.0,
            Scene::Light => 10.0,
        }
    }

    fn escape_radius(self) -> f64 {
        match self {
            Scene::Light => 45.0,
            _ => f64::INFINITY,
        }
    }
}

const ORBIT_P: f64 = 20.0;
const ORBIT_E: f64 = 0.5;

/// Test particles and light moving on exact geodesics of a non-rotating black hole (Schwarzschild),
/// drawn in the orbital plane: the strong-field counterpart of `Planets`. Three scenes (keys 1-3):
/// a precessing orbit next to a Newtonian ghost, the innermost stable circular orbit (ISCO) with
/// a body that spirals in and freezes at the horizon, and a wavefront of light that is bent and
/// partly captured (the shadow).
/// Not modelled: spin (a real black hole rotates, which moves the ISCO and the shadow), what a
/// camera would see (light travel time, redshift, an accretion disc), anything inside the
/// horizon, the mass of the test bodies (no gravitational waves).
pub struct BlackHole {
    scene: Scene,
    bodies: Vec<Body>,
    /// Far-away time in M.
    t: f64,
    /// M per second of real time.
    speed: f64,
}

impl BlackHole {
    pub fn new() -> Self {
        Self::scene(Scene::Orbit)
    }

    fn scene(scene: Scene) -> Self {
        let bodies = match scene {
            Scene::Orbit => {
                let orbit = Body::bound_orbit(ORBIT_P, ORBIT_E, ORANGE, 900);
                let ghost = Body::newtonian(orbit.state.pos, orbit.state.u, GRAY, 70);
                vec![orbit, ghost]
            }
            Scene::Isco => vec![
                Body::circular_orbit(10.0, SKYBLUE, 300),
                Body::circular_orbit(6.0, GREEN, 300),
                // 1% too little angular momentum for the circular orbit at 6 M: it spirals in.
                Body::particle(dvec2(6.0, 0.0), dvec2(0.0, 0.99 / 3f64.sqrt()), RED, 300),
            ],
            Scene::Light => (-RAYS..=RAYS)
                .map(|n| {
                    let mut ray = Body::photon(dvec2(-30.0, 0.4 * n as f64), dvec2(1.0, 0.0), YELLOW, usize::MAX);
                    if ray.angular_momentum.abs() < CRITICAL_IMPACT {
                        ray.color = RED;
                    }
                    ray
                })
                .collect(),
        };
        Self { scene, bodies, t: 0.0, speed: scene.start_speed() }
    }
}

impl Simulation for BlackHole {
    fn step(&mut self, dt: f64) {
        for body in &mut self.bodies {
            body.step(dt, self.scene.escape_radius());
        }
        self.t += dt;
    }

    fn is_over(&self) -> bool {
        false // a scene restarts when its key is pressed again
    }

    fn world_bounds(&self) -> Rect {
        self.scene.bounds()
    }

    fn draw(&self) {
        let width = 0.004 * self.scene.bounds().h;

        // Guide rings at the ISCO and the photon sphere, and the horizon.
        draw_poly_lines(0.0, 0.0, 64, 6.0, 0.0, width / 2.0, DARKGRAY);
        draw_poly_lines(0.0, 0.0, 64, 3.0, 0.0, width / 2.0, DARKGRAY);
        // Not draw_circle: that is a 20-sided polygon, which shows at this size.
        draw_poly(0.0, 0.0, 64, 2.0, 0.0, Color::new(0.1, 0.1, 0.1, 1.0));
        draw_poly_lines(0.0, 0.0, 64, 2.0, 0.0, width, GRAY);

        for body in &self.bodies {
            let pos = body.state.pos.as_vec2();
            let n = body.trail.len();
            let points = body.trail.iter().copied().chain(std::iter::once(pos));
            for (i, (p0, p1)) in points.clone().zip(points.skip(1)).enumerate() {
                let alpha = if self.scene == Scene::Light { 0.6 } else { (i as f32 + 1.0) / n as f32 };
                draw_line(p0.x, p0.y, p1.x, p1.y, width, Color::new(body.color.r, body.color.g, body.color.b, alpha));
            }
            let color = if body.fate == Fate::Captured {
                Color::new(body.color.r * 0.5, body.color.g * 0.5, body.color.b * 0.5, 1.0)
            } else {
                body.color
            };
            draw_circle(pos.x, pos.y, 0.008 * self.scene.bounds().h, color);
        }
    }

    fn dt(&self) -> f64 {
        DT
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn handle_input(&mut self) {
        for (key, scene) in [(KeyCode::Key1, Scene::Orbit), (KeyCode::Key2, Scene::Isco), (KeyCode::Key3, Scene::Light)] {
            if is_key_pressed(key) {
                *self = Self::scene(scene);
            }
        }
        let start = self.scene.start_speed();
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(start * 8.0);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(start / 8.0);
        }
    }

    fn hud(&self) -> Vec<String> {
        let mut lines = vec![
            format!("1 orbit  2 isco  3 light   speed = {} M/s (UP/DOWN)", self.speed),
            format!("t = {:.0} M = {} at Sgr A*", self.t, sgr_a_time(self.t)),
        ];
        match self.scene {
            Scene::Orbit => {
                let orbit = &self.bodies[0];
                lines.push(format!("r = {:.2} M   own clock = {:.0} M", orbit.state.pos.length(), orbit.state.tau));
                if let Some(sweep) = orbit.sweep {
                    let exact = apoapsis_sweep(ORBIT_P, ORBIT_E);
                    lines.push(format!(
                        "precession = {:.2} deg/orbit (exact {:.2})",
                        (sweep - TAU).to_degrees(),
                        (exact - TAU).to_degrees()
                    ));
                }
                lines.push(format!(
                    "drift: E {:+.1e}  L {:+.1e}",
                    orbit.energy_now() / orbit.energy - 1.0,
                    orbit.angular_momentum_now() / orbit.angular_momentum - 1.0
                ));
                lines.push("gray: Newton, same start".to_string());
            }
            Scene::Isco => {
                let falling = &self.bodies[2];
                lines.push(format!(
                    "ISCO r = 6 M: period {:.1} M = {} at Sgr A*",
                    TAU * 6f64.powf(1.5),
                    sgr_a_time(TAU * 6f64.powf(1.5))
                ));
                lines.push(format!(
                    "clock rates: {:.3} (10 M)  {:.3} (6 M)  {:.3} (falling)",
                    self.bodies[0].rate(),
                    self.bodies[1].rate(),
                    falling.rate()
                ));
                lines.push(format!("falling: r = {:.3} M   own clock = {:.1} M", falling.state.pos.length(), falling.state.tau));
            }
            Scene::Light => {
                let count = |fate| self.bodies.iter().filter(|body| body.fate == fate).count();
                lines.push(format!("shadow: b < 3*sqrt(3) M = {:.3} M", CRITICAL_IMPACT));
                lines.push(format!("captured {} escaped {} of {}", count(Fate::Captured), count(Fate::Escaped), self.bodies.len()));
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Advances `body` by `steps` steps of `DT` with no escape radius.
    fn advance(body: &mut Body, steps: usize) {
        for _ in 0..steps {
            body.step(DT, f64::INFINITY);
        }
    }

    fn orbit() -> Body {
        Body::bound_orbit(ORBIT_P, ORBIT_E, WHITE, 100)
    }

    #[test]
    fn orbit_constants_match_closed_forms() {
        let orbit = orbit();
        assert!((orbit.energy - 0.981926222).abs() < 1e-9, "E {}", orbit.energy);
        assert!((orbit.angular_momentum - 4.886777774).abs() < 1e-9, "L {}", orbit.angular_momentum);
        // E^2 = ((p - 2)^2 - 4 e^2) / (p (p - 3 - e^2))
        let (p, e) = (ORBIT_P, ORBIT_E);
        let e2 = ((p - 2.0).powi(2) - 4.0 * e * e) / (p * (p - 3.0 - e * e));
        assert!((orbit.energy * orbit.energy - e2).abs() < 1e-12);

        let isco = Body::circular_orbit(6.0, WHITE, 1);
        assert!((isco.energy - (8.0f64 / 9.0).sqrt()).abs() < 1e-9, "E {}", isco.energy);
        assert!((isco.angular_momentum - 12f64.sqrt()).abs() < 1e-9, "L {}", isco.angular_momentum);

        let photon = Body::photon(dvec2(3.0, 0.0), dvec2(0.0, 1.0), WHITE, 1);
        assert!((photon.angular_momentum - CRITICAL_IMPACT).abs() < 1e-9, "L {}", photon.angular_momentum);
    }

    #[test]
    fn conserves_energy_and_angular_momentum() {
        let mut orbit = orbit();
        let (mut worst_e, mut worst_l): (f64, f64) = (0.0, 0.0);
        for _ in 0..200_000 {
            orbit.step(DT, f64::INFINITY);
            worst_e = worst_e.max((orbit.energy_now() / orbit.energy - 1.0).abs());
            worst_l = worst_l.max((orbit.angular_momentum_now() / orbit.angular_momentum - 1.0).abs());
        }
        assert!(worst_e < 1e-12, "energy drift {worst_e}");
        assert!(worst_l < 1e-11, "angular momentum drift {worst_l}");
    }

    #[test]
    fn strong_field_precession_matches_elliptic_integral() {
        let mut orbit = orbit();
        while orbit.sweep.is_none() {
            orbit.step(DT, f64::INFINITY);
        }
        let sweep = orbit.sweep.unwrap();
        assert!((sweep - apoapsis_sweep(ORBIT_P, ORBIT_E)).abs() < 1e-10, "sweep {sweep}");
        assert!((apoapsis_sweep(ORBIT_P, ORBIT_E) - 7.517047113445).abs() < 1e-10);
    }

    #[test]
    fn mercury_precesses_43_arcsec_per_century() {
        // Mercury's orbit in units of GM_sun/c^2 = 1.4766250 km.
        let e = 0.2056;
        let a = 57.909e6 / 1.4766250;
        let p = a * (1.0 - e * e);
        let mut mercury = Body::bound_orbit(p, e, WHITE, 1);
        let dt = TAU * a.powf(1.5) / 20_000.0;
        while mercury.sweep.is_none() {
            mercury.step(dt, f64::INFINITY);
        }
        let arcsec = (mercury.sweep.unwrap() - TAU) * (180.0 / PI * 3600.0);
        let per_century = arcsec * 36525.0 / 87.969;
        assert!((per_century - 42.980).abs() < 0.001, "{per_century} arcsec per century");
    }

    #[test]
    fn circular_orbit_obeys_keplers_third_law() {
        // Far-away period 2 pi r^1.5, proper-time period that times sqrt(1 - 3/r).
        let mut body = Body::circular_orbit(10.0, WHITE, 1);
        let period = TAU * 10f64.powf(1.5);
        let steps = (period / DT).floor();
        advance(&mut body, steps as usize);
        body.step(period - steps * DT, f64::INFINITY); // the remainder, to end at exactly `period`
        assert!((body.state.pos - dvec2(10.0, 0.0)).length() < 1e-8, "pos {}", body.state.pos);
        let proper = period * 0.7f64.sqrt();
        assert!((body.state.tau - proper).abs() < 1e-8, "tau {} expected {proper}", body.state.tau);
    }

    #[test]
    fn isco_is_the_last_stable_orbit() {
        // (a) Disturbed circular orbit at 8 M sweeps exactly 4 pi between apoapsis passages.
        let mut body = Body::particle(dvec2(8.0, 0.0), dvec2(-1e-5, 1.0 / 5f64.sqrt()), WHITE, 1);
        while body.sweep.is_none() {
            body.step(DT, f64::INFINITY);
        }
        assert!((body.sweep.unwrap() - 2.0 * TAU).abs() < 1e-6, "sweep {}", body.sweep.unwrap());

        // (b) Just outside the ISCO the orbit stays circular for a long time.
        let mut outside = Body::particle(dvec2(6.05, 0.0), dvec2(-1e-5, 1.0 / 3.05f64.sqrt()), WHITE, 1);
        for _ in 0..100_000 {
            outside.step(DT, f64::INFINITY);
            let off = (outside.state.pos.length() - 6.05).abs();
            assert!(off < 0.002, "drifted {off} from r = 6.05");
        }

        // (c) Just inside, it plunges.
        let mut inside = Body::particle(dvec2(5.95, 0.0), dvec2(-1e-5, 1.0 / 2.95f64.sqrt()), WHITE, 1);
        let mut t = 0.0;
        while inside.fate == Fate::Moving && t < 1500.0 {
            inside.step(DT, f64::INFINITY);
            t += DT;
        }
        assert!(inside.fate == Fate::Captured, "not captured by t = {t}");
    }

    #[test]
    fn photon_sphere_orbit() {
        // Light on the unstable circular orbit at r = 3 for one period 6 pi sqrt(3).
        let mut photon = Body::photon(dvec2(3.0, 0.0), dvec2(0.0, 1.0), WHITE, 1);
        for _ in 0..653 {
            photon.step(DT, f64::INFINITY);
            assert!((photon.state.pos.length() - 3.0).abs() < 1e-5, "r {}", photon.state.pos.length());
        }
    }

    #[test]
    fn critical_impact_parameter_is_3_sqrt_3() {
        // Bisect the starting height of a ray: captured below, escaping above.
        let (mut low, mut high) = (5.0, 5.4);
        let mut boundary = 0.0;
        for _ in 0..40 {
            let y = (low + high) / 2.0;
            let mut ray = Body::photon(dvec2(-100.0, y), dvec2(1.0, 0.0), WHITE, 1);
            while ray.fate == Fate::Moving {
                ray.step(DT, 110.0);
            }
            if ray.fate == Fate::Captured { low = y } else { high = y }
            boundary = ray.angular_momentum.abs();
        }
        assert!((boundary - CRITICAL_IMPACT).abs() < 1e-8, "boundary L {boundary}");
    }

    #[test]
    fn light_grazing_the_sun_bends_1_75_arcsec() {
        let b = 695_700.0 / 1.4766250;
        let mut ray = Body::photon(dvec2(-1e9, b), dvec2(1.0, 0.0), WHITE, 1);
        while ray.state.pos.x < 1e9 {
            ray.step(1e4, f64::INFINITY);
        }
        let bend = -ray.state.u.y.atan2(ray.state.u.x) * (180.0 / PI * 3600.0);
        assert!((bend - 1.751201).abs() < 1e-5, "deflection {bend} arcsec");
    }

    #[test]
    fn radial_light_shows_the_shapiro_delay() {
        // Far-away time to fall from r = 40 to r: (40 - r) + 2 ln(38 / (r - 2)); flat space has 40 - r.
        let mut ray = Body::photon(dvec2(40.0, 0.0), dvec2(-1.0, 0.0), WHITE, 1);
        let mut t = 0.0;
        while ray.state.pos.length() > 3.0 {
            ray.step(DT, f64::INFINITY);
            t += DT;
            let r = ray.state.pos.length();
            let expected = (40.0 - r) + 2.0 * (38.0 / (r - 2.0)).ln();
            assert!((t - expected).abs() < 1e-8, "t {t} expected {expected} at r {r}");
        }
        assert!((t - 44.30).abs() < 0.06, "final t {t} (flat space: 37)");
    }

    #[test]
    fn free_fall_freezes_at_the_horizon() {
        // Dropped from rest at infinity (E = 1), the proper time is finite while the far-away
        // time to reach the horizon is not: the body is "captured" at 2.01.
        let mut body = Body::particle(dvec2(20.0, 0.0), dvec2(-(0.1f64).sqrt(), 0.0), WHITE, 1);
        assert!((body.energy - 1.0).abs() < 1e-12, "E {}", body.energy);
        let mut t = 0.0;
        while body.fate == Fate::Moving {
            body.step(DT, f64::INFINITY);
            t += DT;
            let r = body.state.pos.length();
            let expected = 2.0 / 3.0 * (20f64.powf(1.5) - r.powf(1.5)) / 2f64.sqrt();
            assert!((body.state.tau - expected).abs() < 1e-9, "tau {} expected {expected}", body.state.tau);
        }
        assert!((t - 61.55).abs() < 0.06, "captured at t = {t}");
        assert!((body.state.tau - 40.8205).abs() < 1e-3, "tau at capture {}", body.state.tau);
    }

    #[test]
    fn shadow_is_the_rays_below_the_critical_impact_parameter() {
        let mut scene = BlackHole::scene(Scene::Light);
        for _ in 0..2400 {
            scene.step(DT);
        }
        for ray in &scene.bodies {
            assert!(ray.fate != Fate::Moving, "ray with L = {} still moving", ray.angular_momentum);
            let captured = ray.angular_momentum.abs() < CRITICAL_IMPACT;
            assert_eq!(ray.fate == Fate::Captured, captured, "ray with L = {}", ray.angular_momentum);
        }
        let captured = scene.bodies.iter().filter(|ray| ray.fate == Fate::Captured).count();
        assert_eq!(captured, 25);
    }
}
