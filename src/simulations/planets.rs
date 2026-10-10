use std::collections::VecDeque;

use macroquad::prelude::*;

use crate::simulation::Simulation;

/// Units: distance in AU, time in years, mass in solar masses.
/// In these units G * (1 solar mass) = 4 pi^2 (Earth's orbit has a = 1 AU, T = 1 yr).
const G: f64 = 4.0 * std::f64::consts::PI * std::f64::consts::PI;

/// Physics step in years. Mercury's orbit is 0.24 yr, so this is ~480 steps per orbit.
const DT: f64 = 0.0005;
/// Simulated years per second of real time at the start; adjustable with UP/DOWN.
const START_SPEED: f64 = 0.5;
const MIN_SPEED: f64 = START_SPEED / 16.0;
const MAX_SPEED: f64 = START_SPEED * 32.0;

// Drawing. See `to_display` for how AU map to the screen.
const STAR_RADIUS: f32 = 0.25;
const PLANET_RADIUS: f32 = 0.09;
const MOON_RADIUS: f32 = 0.05;
/// The Moon is only 0.0026 AU from Earth, which would sit on top of it. Its offset from Earth is
/// drawn this many times larger (it is not subject to the sqrt scaling of the planets' orbits).
const MOON_EXAGGERATION: f64 = 100.0;
const TRAIL_WIDTH: f32 = 0.03;
/// A trail point is added each time a planet has moved this far around the star (rad)...
const TRAIL_ANGLE: f64 = 0.05;
/// ...and the trail keeps the last ~2*pi rad, i.e. roughly one orbit.
const TRAIL_LEN: usize = 130;

/// Orbital elements at J2000 (planets: Standish, JPL "Keplerian elements for approximate
/// positions"), reduced to the plane of the ecliptic: orbits are 2D here, so inclination is ignored.
struct Elements {
    /// Mass in solar masses.
    m: f64,
    /// Semi-major axis, AU.
    a: f64,
    /// Eccentricity.
    e: f64,
    /// Mean longitude at J2000, degrees.
    mean_longitude: f64,
    /// Longitude of perihelion (perigee for the Moon) at J2000, degrees.
    perihelion: f64,
    color: Color,
}

const STAR_MASS: f64 = 1.0;

const PLANETS: [Elements; 8] = [
    Elements { m: 1.6601e-7, a: 0.38710, e: 0.20563, mean_longitude: 252.25, perihelion: 77.46, color: LIGHTGRAY }, // Mercury
    Elements { m: 2.4478e-6, a: 0.72333, e: 0.00677, mean_longitude: 181.98, perihelion: 131.53, color: BEIGE }, // Venus
    // The Earth-Moon system: a = 1 AU is the orbit of its barycenter. It is split into Earth and Moon.
    Elements { m: 3.0404e-6, a: 1.00000, e: 0.01671, mean_longitude: 100.46, perihelion: 102.94, color: GREEN }, // Earth
    Elements { m: 3.2272e-7, a: 1.52368, e: 0.09340, mean_longitude: 355.45, perihelion: 336.04, color: RED }, // Mars
    Elements { m: 9.5479e-4, a: 5.20260, e: 0.04839, mean_longitude: 34.40, perihelion: 14.75, color: ORANGE }, // Jupiter
    Elements { m: 2.8589e-4, a: 9.55490, e: 0.05551, mean_longitude: 49.94, perihelion: 92.43, color: GOLD }, // Saturn
    Elements { m: 4.3662e-5, a: 19.21840, e: 0.04630, mean_longitude: 313.23, perihelion: 170.96, color: SKYBLUE }, // Uranus
    Elements { m: 5.1514e-5, a: 30.11040, e: 0.00899, mean_longitude: 304.88, perihelion: 44.97, color: VIOLET }, // Neptune
];

/// Index in `PLANETS` of the Earth-Moon system.
const EARTH: usize = 2;

/// The Moon's orbit around Earth (geocentric elements, JPL/Meeus).
const MOON: Elements = Elements { m: 3.694e-8, a: 0.00257, e: 0.0549, mean_longitude: 218.32, perihelion: 83.35, color: WHITE };

struct Body {
    pos: DVec2,
    v: DVec2,
    a: DVec2,
    m: f64,
    color: Color,
    radius: f32,
    /// Index of the body this one orbits (the star for planets, Earth for the Moon).
    center: usize,
    /// Recent positions relative to `center`, for drawing. Empty for the star itself.
    trail: VecDeque<DVec2>,
}

/// The Sun, the eight planets and the Moon, all attracting each other (Newtonian N-body).
pub struct Planets {
    /// The star is `bodies[0]`.
    bodies: Vec<Body>,
    /// Total energy at the start, to measure drift against.
    energy0: f64,
    speed: f64,
    t: f64,
}

/// Position and velocity relative to the body orbited, from orbital elements.
/// `mu` is G times the sum of the two masses.
fn initial_state(mu: f64, orbit: &Elements) -> (DVec2, DVec2) {
    let (e, a) = (orbit.e, orbit.a);
    let mean_anomaly = (orbit.mean_longitude - orbit.perihelion).to_radians();

    // Kepler's equation M = E - e sin(E), by Newton iteration.
    let mut big_e = mean_anomaly;
    for _ in 0..20 {
        big_e -= (big_e - e * big_e.sin() - mean_anomaly) / (1.0 - e * big_e.cos());
    }

    // In the orbit's own frame, with perihelion along +x (counter-clockwise motion).
    let r = a * (1.0 - e * big_e.cos());
    let k = (mu * a).sqrt() / r;
    let b = (1.0 - e * e).sqrt();
    let pos = dvec2(a * (big_e.cos() - e), a * b * big_e.sin());
    let v = dvec2(-k * big_e.sin(), k * b * big_e.cos());

    // Rotate so perihelion points along its longitude.
    let rotation = DMat2::from_angle(orbit.perihelion.to_radians());
    (rotation * pos, rotation * v)
}

impl Planets {
    pub fn new() -> Self {
        let mut bodies = vec![Body {
            pos: DVec2::ZERO,
            v: DVec2::ZERO,
            a: DVec2::ZERO,
            m: STAR_MASS,
            color: YELLOW,
            radius: STAR_RADIUS,
            center: 0,
            trail: VecDeque::new(),
        }];
        for planet in &PLANETS {
            let (pos, v) = initial_state(G * (STAR_MASS + planet.m), planet);
            let (m, color, radius) = (planet.m, planet.color, PLANET_RADIUS);
            bodies.push(Body { pos, v, a: DVec2::ZERO, m, color, radius, center: 0, trail: VecDeque::new() });
        }

        // The Earth-Moon system's barycenter follows the orbit above; put Earth and the Moon
        // around it, at the Moon's orbit about Earth.
        let earth = EARTH + 1; // bodies[0] is the star
        let system_m = bodies[earth].m;
        let (rel_pos, rel_v) = initial_state(G * system_m, &MOON);
        let (moon_share, earth_share) = (MOON.m / system_m, 1.0 - MOON.m / system_m);
        bodies.push(Body {
            pos: bodies[earth].pos + rel_pos * earth_share,
            v: bodies[earth].v + rel_v * earth_share,
            a: DVec2::ZERO,
            m: MOON.m,
            color: MOON.color,
            radius: MOON_RADIUS,
            center: earth,
            trail: VecDeque::new(),
        });
        bodies[earth].pos -= rel_pos * moon_share;
        bodies[earth].v -= rel_v * moon_share;
        bodies[earth].m = system_m - MOON.m;

        // The bodies were set up relative to a star at rest, which leaves the system with net
        // momentum. Move to the center-of-mass frame so the star wobbles instead of drifting away.
        let total_m: f64 = bodies.iter().map(|body| body.m).sum();
        let center: DVec2 = bodies.iter().map(|body| body.m * body.pos).sum::<DVec2>() / total_m;
        let drift: DVec2 = bodies.iter().map(|body| body.m * body.v).sum::<DVec2>() / total_m;
        for body in &mut bodies {
            body.pos -= center;
            body.v -= drift;
        }

        let mut planets = Self { bodies, energy0: 0.0, speed: START_SPEED, t: 0.0 };
        planets.update_accelerations();
        planets.energy0 = planets.energy();
        planets
    }

    /// Sets every body's acceleration from the gravity of all the others.
    fn update_accelerations(&mut self) {
        for body in &mut self.bodies {
            body.a = DVec2::ZERO;
        }
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                // Newton's law: a_i = G m_j d / |d|^3, and the opposite on j (third law).
                let d = self.bodies[j].pos - self.bodies[i].pos;
                let k = G / (d.length_squared() * d.length());
                let (m_i, m_j) = (self.bodies[i].m, self.bodies[j].m);
                self.bodies[i].a += d * (k * m_j);
                self.bodies[j].a -= d * (k * m_i);
            }
        }
    }

    /// Kinetic plus gravitational potential energy.
    fn energy(&self) -> f64 {
        let mut e = 0.0;
        for (i, body) in self.bodies.iter().enumerate() {
            e += 0.5 * body.m * body.v.length_squared();
            for other in &self.bodies[(i + 1)..] {
                e -= G * body.m * other.m / (other.pos - body.pos).length();
            }
        }
        e
    }

    /// Total momentum. Starts at zero in the center-of-mass frame and must stay there.
    fn momentum(&self) -> DVec2 {
        self.bodies.iter().map(|body| body.m * body.v).sum()
    }
}

/// Maps a position relative to the star (AU) to drawing coordinates. The radius is square-rooted,
/// since drawing Mercury (0.39 AU) and Neptune (30 AU) to scale won't fit on one screen.
fn to_display(rel: DVec2) -> Vec2 {
    (rel.normalize_or_zero() * rel.length().sqrt()).as_vec2()
}

impl Planets {
    /// Drawing coordinates of a point given relative to the body `center`.
    fn display_point(&self, center: usize, rel: DVec2) -> Vec2 {
        let star = self.bodies[0].pos;
        if center == 0 {
            return to_display(rel);
        }
        // Around a planet (the Moon): exaggerated offset from the planet's drawn position.
        to_display(self.bodies[center].pos - star) + (rel * MOON_EXAGGERATION).as_vec2()
    }
}

impl Simulation for Planets {
    fn step(&mut self, dt: f64) {
        // Velocity Verlet (kick-drift-kick): second order, symplectic, one force evaluation.
        for body in &mut self.bodies {
            body.v += body.a * (dt / 2.0);
            body.pos += body.v * dt;
        }
        self.update_accelerations();
        for body in &mut self.bodies {
            body.v += body.a * (dt / 2.0);
        }
        self.t += dt;

        // Trails are sampled by angle around the body orbited, so each shows about one orbit.
        let centers: Vec<DVec2> = self.bodies.iter().map(|body| body.pos).collect();
        for body in &mut self.bodies[1..] {
            let rel = body.pos - centers[body.center];
            let moved = match body.trail.back() {
                Some(last) => (rel.to_angle() - last.to_angle() + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI,
                None => f64::INFINITY,
            };
            if moved.abs() >= TRAIL_ANGLE {
                if body.trail.len() == TRAIL_LEN {
                    body.trail.pop_front();
                }
                body.trail.push_back(rel);
            }
        }
    }

    fn is_over(&self) -> bool {
        false // runs indefinitely
    }

    fn dt(&self) -> f64 {
        DT
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
    }

    fn world_bounds(&self) -> Rect {
        // Neptune is at sqrt(30) ~ 5.5 in display coordinates.
        Rect::new(-8.55, -5.7, 17.1, 11.4)
    }

    fn draw(&self) {
        // Everything is drawn relative to the star, so the star itself stays in the middle.
        draw_circle(0.0, 0.0, STAR_RADIUS, YELLOW);

        for body in &self.bodies[1..] {
            // Trail: fade from transparent to the body's color.
            let n = body.trail.len();
            for (i, (p0, p1)) in body.trail.iter().zip(body.trail.iter().skip(1)).enumerate() {
                let alpha = (i as f32 + 1.0) / n as f32;
                let c = Color::new(body.color.r, body.color.g, body.color.b, alpha * 0.8);
                let (p0, p1) = (self.display_point(body.center, *p0), self.display_point(body.center, *p1));
                draw_line(p0.x, p0.y, p1.x, p1.y, TRAIL_WIDTH, c);
            }
            let rel = body.pos - self.bodies[body.center].pos;
            let p = self.display_point(body.center, rel);
            draw_circle(p.x, p.y, body.radius, body.color);
        }
    }

    fn hud(&self) -> Vec<String> {
        vec![
            format!("t = {:.1} yr   speed = {:.2} yr/s (UP/DOWN)", self.t, self.speed),
            format!("energy drift = {:+.1e}", (self.energy() - self.energy0) / self.energy0.abs()),
            format!("|momentum| = {:.1e} Msun AU/yr", self.momentum().length()),
            "distances: sqrt(AU) from the Sun; Moon-Earth x100".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conserves_energy_and_momentum() {
        let mut planets = Planets::new();
        assert!(planets.momentum().length() < 1e-12);

        // 20 years: ~1.7 Jupiter orbits, 83 Mercury orbits.
        for _ in 0..(20.0 / DT) as usize {
            planets.step(DT);
        }
        let drift = (planets.energy() - planets.energy0) / planets.energy0.abs();
        assert!(drift.abs() < 1e-6, "energy drift {drift}");
        assert!(planets.momentum().length() < 1e-12, "momentum {}", planets.momentum().length());
    }

    #[test]
    fn moon_stays_in_orbit_around_earth() {
        let mut planets = Planets::new();
        let moon = planets.bodies.len() - 1;
        let earth = EARTH + 1;
        // Lunar distance is 0.0024-0.0027 AU in reality (perigee-apogee); allow some margin.
        for _ in 0..(5.0 / DT) as usize {
            planets.step(DT);
            let d = (planets.bodies[moon].pos - planets.bodies[earth].pos).length();
            assert!(d > 0.0022 && d < 0.0029, "Moon-Earth distance {d} AU");
        }
    }

    #[test]
    fn planets_start_at_their_semi_major_axes() {
        // Distance from the star should be within a(1 +- e) of the real orbit.
        let planets = Planets::new();
        for (body, planet) in planets.bodies[1..].iter().zip(&PLANETS) {
            let r = (body.pos - planets.bodies[0].pos).length();
            let (min, max) = (planet.a * (1.0 - planet.e), planet.a * (1.0 + planet.e));
            assert!(r > min * 0.999 && r < max * 1.001, "a = {}: r = {r}", planet.a);
        }
    }
}
