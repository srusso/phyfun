use std::collections::VecDeque;

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: distance in AU, time in Julian years (365.25 days), mass in solar masses.
const AU_KM: f64 = 149_597_870.7;
const YEAR_S: f64 = 365.25 * 86_400.0;
const SUN_GM: f64 = 132_712_440_041.279_419; // km^3/s^2
/// G * (1 solar mass) in AU^3/yr^2, so with masses in solar masses `G` is the usual constant.
/// This is close to 4 pi^2 but not equal: the real Earth year isn't exactly 1 Julian year.
const G: f64 = SUN_GM * YEAR_S * YEAR_S / (AU_KM * AU_KM * AU_KM);
const DAYS_PER_YEAR: f64 = 365.25;
/// Speed of light in AU/yr (299792.458 km/s).
const C: f64 = 299_792.458 * YEAR_S / AU_KM;

/// Earth's oblateness (equatorial bulge), which pulls on the Moon: the J2 coefficient and the
/// equatorial radius (km) from the DE441 header, the axis tilt from the ecliptic (arcseconds), and
/// how far the axis precesses around the ecliptic's pole per century (arcseconds; IAU 2006
/// precession on the fixed J2000 ecliptic, from memory). The precession moves the Moon by 0.3 km
/// over a century.
const EARTH_J2: f64 = 1.082_625_39e-3;
const EARTH_RADIUS_KM: f64 = 6378.1366;
const EARTH_TILT_ARCSEC: f64 = 84_381.448;
const EARTH_PRECESSION_ARCSEC: f64 = 5038.481_507;

/// Physics step in years. Mercury's orbit is 0.24 yr, so this is ~480 steps per orbit, and
/// the Moon's (0.075 yr) ~150. With the sixth-order integrator this leaves positions accurate to
/// ~1e-8 AU over a century, far below the errors of the model itself.
const DT: f64 = 0.0005;

/// Yoshida's sixth-order symplectic integrator (1990, solution A): one step is seven leapfrog
/// steps of sizes `weight * dt`. The weights sum to 1, and the negative one steps backwards in time.
const YOSHIDA6: [f64; 7] = {
    const W1: f64 = -1.177_679_984_178_87;
    const W2: f64 = 0.235_573_213_359_357;
    const W3: f64 = 0.784_513_610_477_560;
    const W0: f64 = 1.0 - 2.0 * (W1 + W2 + W3);
    [W3, W2, W1, W0, W1, W2, W3]
};

/// Simulated years per second of real time at the start; adjustable with UP/DOWN.
const START_SPEED: f64 = 0.5;
const MIN_SPEED: f64 = START_SPEED / 16.0;
const MAX_SPEED: f64 = START_SPEED * 32.0;

// Drawing. See `to_display` for how AU map to the screen.
const STAR_RADIUS: f32 = 0.25;
const PLANET_RADIUS: f32 = 0.09;
const MOON_RADIUS: f32 = 0.05;
const TRAIL_WIDTH: f32 = 0.03;
/// The Moon is only 0.0026 AU from Earth, which would sit on top of it. Its offset from Earth is
/// drawn this many times larger (it is not subject to the sqrt scaling of the planets' orbits).
const MOON_EXAGGERATION: f64 = 100.0;
/// A trail point is added each time a body has moved this far around the one it orbits (rad)...
const TRAIL_ANGLE: f64 = 0.05;
/// ...and the trail keeps the last ~2*pi rad, i.e. roughly one orbit.
const TRAIL_LEN: usize = 130;

/// Initial state of a body, from JPL Horizons (DE441 ephemeris) at J2000: 2000-01-01 12:00 TDB.
/// Positions and velocities are relative to the solar system barycenter, in the ecliptic frame.
/// Mars and the giant planets are the barycenters of their systems (planet plus moons), with the
/// mass of the whole system, so their moons are accounted for without simulating them.
struct Start {
    /// G times mass, km^3/s^2 (JPL's DE440/441 values).
    gm: f64,
    /// AU.
    pos: [f64; 3],
    /// AU/day.
    v: [f64; 3],
    /// Index in `START` of the body this one is drawn around (the Sun, or Earth for the Moon).
    center: usize,
    color: Color,
}

const EARTH: usize = 3;
const MOON: usize = 9;

/// The Sun, the eight planets and the Moon, at the indices above.
const START: [Start; 10] = [
    Start { gm: 132712440041.279419, pos: [-0.007137179161607904, -0.002795997495567612, 0.0002062985061910257], v: [5.378460339618783e-06, -7.406916207973516e-06, -9.43429343213747e-08], center: 0, color: YELLOW }, // Sun
    Start { gm: 22031.868551, pos: [-0.1372307845370601, -0.450083615630924, -0.02439200845186075], v: [0.02137177412835578, -0.006455396660044872, -0.002487957757771483], center: 0, color: LIGHTGRAY }, // Mercury
    Start { gm: 324858.592, pos: [-0.725439475506997, -0.03545030569537367, 0.04122048053303724], v: [0.0008034959761149407, -0.02030262561844178, -0.0003235458586621571], center: 0, color: BEIGE }, // Venus
    Start { gm: 398600.435507, pos: [-0.1842722784343177, 0.964445689270963, 0.0002022132246085051], v: [-0.01720224660838933, -0.003166189060532839, 1.064592514002072e-08], center: 0, color: GREEN }, // Earth
    Start { gm: 42828.375816, pos: [1.383578742584679, -0.01621231564790753, -0.03426136426966323], v: [0.0006768779813732038, 0.01517984117494233, 0.0003015574196696519], center: 0, color: RED }, // Mars
    Start { gm: 126712764.1, pos: [3.994039981964449, 2.935780109470535, -0.1015789781275531], v: [-0.004562935066413099, 0.006435799094013335, 7.548517732342305e-05], center: 0, color: ORANGE }, // Jupiter
    Start { gm: 37940584.8418, pos: [6.399271680371201, 6.567193619416989, -0.3688701277386623], v: [-0.004286973413503631, 0.003882908781806891, 0.0001028535064646593], center: 0, color: GOLD }, // Saturn
    Start { gm: 5794556.4, pos: [14.42471943522018, -13.73711746626653, -0.2379353892930592], v: [0.002683484051759769, 0.002665288979504713, -2.486640648937235e-05], center: 0, color: SKYBLUE }, // Uranus
    Start { gm: 6836527.10058, pos: [16.80490978889128, -24.99455888678026, 0.1274291784264387], v: [0.002584652719387356, 0.001769493903808153, -9.600395792023743e-05], center: 0, color: VIOLET }, // Neptune
    Start { gm: 4902.800118, pos: [-0.1862215600840044, 0.9626075632308897, 0.000444671198490556], v: [-0.01683057613107253, -0.003588367637063646, -6.634893538849906e-06], center: EARTH, color: WHITE }, // Moon
];

struct Body {
    pos: DVec3,
    v: DVec3,
    a: DVec3,
    m: f64,
    color: Color,
    radius: f32,
    /// Index of the body this one orbits, for drawing (the Sun for planets, Earth for the Moon).
    center: usize,
    /// Recent positions relative to `center`, seen from above the ecliptic. Empty for the Sun.
    trail: VecDeque<DVec2>,
}

/// The Sun, the eight planets and the Moon, all attracting each other (Newtonian N-body), plus
/// the first post-Newtonian relativity between every pair of bodies (the Einstein-Infeld-Hoffmann
/// equations, which is what makes Mercury's orbit precess) and the pull of Earth's equatorial
/// bulge on the Moon.
/// Not modelled: asteroids, Pluto and the Kuiper belt, and the Moon's own shape and tides.
/// Motion is 3D, so inclinations are real, but it is drawn from above the ecliptic.
pub struct Planets {
    /// The Sun is `bodies[0]`.
    bodies: Vec<Body>,
    /// Total energy at the start, to measure drift against.
    energy0: f64,
    speed: f64,
    /// Years since J2000.
    t: f64,
}

impl Planets {
    pub fn new() -> Self {
        let mut bodies: Vec<Body> = START
            .iter()
            .enumerate()
            .map(|(i, start)| {
                let radius = match i {
                    0 => STAR_RADIUS,
                    MOON => MOON_RADIUS,
                    _ => PLANET_RADIUS,
                };
                Body {
                    pos: DVec3::from(start.pos),
                    v: DVec3::from(start.v) * DAYS_PER_YEAR,
                    a: DVec3::ZERO,
                    m: start.gm / SUN_GM,
                    color: start.color,
                    radius,
                    center: start.center,
                    trail: VecDeque::new(),
                }
            })
            .collect();

        // JPL's frame is centered on the real solar system's barycenter, which also counts the
        // asteroids, Pluto and so on. Move to the center of mass of the bodies simulated here, so
        // that total momentum is exactly zero and the Sun's wobble doesn't drift away.
        let total_m: f64 = bodies.iter().map(|body| body.m).sum();
        let center: DVec3 = bodies.iter().map(|body| body.m * body.pos).sum::<DVec3>() / total_m;
        let drift: DVec3 = bodies.iter().map(|body| body.m * body.v).sum::<DVec3>() / total_m;
        for body in &mut bodies {
            body.pos -= center;
            body.v -= drift;
        }

        let mut planets = Self { bodies, energy0: 0.0, speed: START_SPEED, t: 0.0 };
        planets.update_accelerations();
        // With relativity the conserved momentum is not just the sum of m v. Remove what is left
        // of it (twice, since it depends on the velocities themselves), or it starts at 1.5e-13.
        for _ in 0..2 {
            let drift = planets.momentum() / total_m;
            for body in &mut planets.bodies {
                body.v -= drift;
            }
        }
        planets.energy0 = planets.energy();
        planets
    }

    /// Direction of Earth's axis in the ecliptic frame. It is tilted by `EARTH_TILT_ARCSEC` and
    /// turns about the ecliptic's pole (clockwise seen from above) by the precession since J2000.
    fn earth_pole(&self) -> DVec3 {
        let tilt = (EARTH_TILT_ARCSEC / 3600.0).to_radians();
        let turned = (EARTH_PRECESSION_ARCSEC / 3600.0).to_radians() * self.t / 100.0;
        dvec3(tilt.sin() * turned.sin(), tilt.sin() * turned.cos(), tilt.cos())
    }

    /// Sets every body's acceleration from the gravity of all the others.
    fn update_accelerations(&mut self) {
        for body in &mut self.bodies {
            body.a = DVec3::ZERO;
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

        // Extra pull of Earth's bulge on the Moon:
        //   a = -3/2 J2 G m_earth R^2 / r^5 * [(1 - 5 z^2 / r^2) r_vec + 2 z p],
        // with r_vec the Moon's position relative to Earth, p Earth's axis and z = r_vec . p.
        let pole = self.earth_pole();
        let radius = EARTH_RADIUS_KM / AU_KM;
        let (earth, moon) = (&self.bodies[EARTH], &self.bodies[MOON]);
        let rel = moon.pos - earth.pos;
        let (r2, z) = (rel.length_squared(), rel.dot(pole));
        let k = -1.5 * EARTH_J2 * G * earth.m * radius * radius / (r2 * r2 * r2.sqrt());
        let a_moon = (rel * (1.0 - 5.0 * z * z / r2) + pole * (2.0 * z)) * k;
        let moon_to_earth_mass = moon.m / earth.m;
        self.bodies[MOON].a += a_moon;
        self.bodies[EARTH].a -= a_moon * moon_to_earth_mass;
    }

    /// One velocity Verlet (kick-drift-kick) step: second order, time-symmetric, one force evaluation.
    fn leapfrog(&mut self, dt: f64) {
        self.kick(dt / 2.0);
        for body in &mut self.bodies {
            body.pos += body.v * dt;
        }
        self.update_accelerations();
        self.kick(dt / 2.0);
    }

    /// Changes every velocity by `h` times its acceleration (Newton, and Earth's bulge).
    fn kick(&mut self, h: f64) {
        for body in &mut self.bodies {
            body.v += body.a * h;
        }
    }

    /// Newtonian potential at each body, U_i = sum over the others of G m_k / r_ik.
    fn potentials(&self) -> Vec<f64> {
        let mut u = vec![0.0; self.bodies.len()];
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                let r = (self.bodies[j].pos - self.bodies[i].pos).length();
                u[i] += G * self.bodies[j].m / r;
                u[j] += G * self.bodies[i].m / r;
            }
        }
        u
    }

    /// Accelerations that are small (about 1e-8 of the total) and expensive: first
    /// post-Newtonian relativity between every pair of bodies (the Einstein-Infeld-Hoffmann
    /// equations). For i != j, with `d` = position of j minus i, `a_j` the Newtonian
    /// acceleration of j and U the potentials from `potentials`:
    ///   B = -4 U_i - U_j + v_i^2 + 2 v_j^2 - 4 v_i.v_j - 3/2 (d.v_j / r)^2 + 1/2 d.a_j
    ///   a_i += G m_j / c^2 [ (B d - (d.(4 v_i - 3 v_j)) (v_i - v_j)) / r^3 + 7/2 a_j / r ]
    /// Uses the positions, velocities and `body.a` as they are now.
    fn slow_accelerations(&self) -> Vec<DVec3> {
        let u = self.potentials();
        let mut a = vec![DVec3::ZERO; self.bodies.len()];
        for (i, bi) in self.bodies.iter().enumerate() {
            for (j, bj) in self.bodies.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = bj.pos - bi.pos;
                let r = d.length();
                let b = -4.0 * u[i] - u[j] + bi.v.length_squared() + 2.0 * bj.v.length_squared()
                    - 4.0 * bi.v.dot(bj.v)
                    - 1.5 * (d.dot(bj.v) / r).powi(2)
                    + 0.5 * d.dot(bj.a);
                let along = d.dot(4.0 * bi.v - 3.0 * bj.v);
                a[i] += (G * bj.m / (C * C)) * ((d * b - (bi.v - bj.v) * along) / (r * r * r) + bj.a * (3.5 / r));
            }
        }
        a
    }

    /// Changes every velocity by `h` times the slow accelerations.
    fn slow_kick(&mut self, h: f64) {
        let a = self.slow_accelerations();
        for (body, a) in self.bodies.iter_mut().zip(a) {
            body.v += a * h;
        }
    }

    /// Total energy including the post-Newtonian terms of the EIH equations (conserved to ~1e-13):
    ///   E = sum_i m_i [v^2/2 - U/2 + (3 v^4/8 + 3 U v^2/2 + U^2/2) / c^2]
    ///       - G / (4 c^2) sum_i sum_{j != i} (m_i m_j / r) (7 v_i.v_j + (n.v_i)(n.v_j)),
    /// with n the unit vector between i and j. Earth's bulge is left out; it doesn't show at this level.
    fn energy(&self) -> f64 {
        let u = self.potentials();
        let c2 = C * C;
        let mut e = 0.0;
        for (i, bi) in self.bodies.iter().enumerate() {
            let v2 = bi.v.length_squared();
            e += bi.m * (v2 / 2.0 - u[i] / 2.0 + (0.375 * v2 * v2 + 1.5 * u[i] * v2 + 0.5 * u[i] * u[i]) / c2);
            for (j, bj) in self.bodies.iter().enumerate() {
                if i != j {
                    let d = bj.pos - bi.pos;
                    let r = d.length();
                    let n = d / r;
                    e -= G / (4.0 * c2) * bi.m * bj.m / r * (7.0 * bi.v.dot(bj.v) + n.dot(bi.v) * n.dot(bj.v));
                }
            }
        }
        e
    }

    /// Total momentum with the post-Newtonian terms. Starts at zero in the center-of-mass frame
    /// and must stay there:
    ///   P = sum_i m_i v_i [1 + (v^2/2 - U/2) / c^2] - G / (2 c^2) sum_i sum_{j != i} (m_i m_j / r)(n.v_j) n
    fn momentum(&self) -> DVec3 {
        let u = self.potentials();
        let c2 = C * C;
        let mut p = DVec3::ZERO;
        for (i, bi) in self.bodies.iter().enumerate() {
            p += bi.m * bi.v * (1.0 + (bi.v.length_squared() / 2.0 - u[i] / 2.0) / c2);
            for (j, bj) in self.bodies.iter().enumerate() {
                if i != j {
                    let d = bj.pos - bi.pos;
                    let r = d.length();
                    let n = d / r;
                    p -= G / (2.0 * c2) * bi.m * bj.m / r * n.dot(bj.v) * n;
                }
            }
        }
        p
    }

    /// Drawing coordinates of a point given (seen from above) relative to the body `center`.
    fn display_point(&self, center: usize, rel: DVec2) -> Vec2 {
        if center == 0 {
            return to_display(rel);
        }
        // Around a planet (the Moon): exaggerated offset from the planet's drawn position.
        let planet = (self.bodies[center].pos - self.bodies[0].pos).truncate();
        to_display(planet) + (rel * MOON_EXAGGERATION).as_vec2()
    }
}

/// Maps a position relative to the Sun (AU) to drawing coordinates. The radius is square-rooted,
/// since drawing Mercury (0.39 AU) and Neptune (30 AU) to scale won't fit on one screen.
fn to_display(rel: DVec2) -> Vec2 {
    (rel.normalize_or_zero() * rel.length().sqrt()).as_vec2()
}

impl Simulation for Planets {
    fn step(&mut self, dt: f64) {
        // The slow accelerations are applied once at each end of the step, at second order; they
        // are 1e-8 of the total, so that costs nothing visible and saves five sixths of the work.
        self.slow_kick(dt / 2.0);
        for weight in YOSHIDA6 {
            self.leapfrog(weight * dt);
        }
        self.slow_kick(dt / 2.0);
        self.t += dt;

        // Trails are sampled by angle around the body orbited, so each shows about one orbit.
        let centers: Vec<DVec3> = self.bodies.iter().map(|body| body.pos).collect();
        for body in &mut self.bodies[1..] {
            let rel = (body.pos - centers[body.center]).truncate();
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

    fn world_bounds(&self) -> Rect {
        // Neptune is at sqrt(30) ~ 5.5 in display coordinates.
        Rect::new(-8.55, -5.7, 17.1, 11.4)
    }

    fn draw(&self) {
        // Everything is drawn relative to the Sun, so the Sun itself stays in the middle.
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
            let rel = (body.pos - self.bodies[body.center].pos).truncate();
            let p = self.display_point(body.center, rel);
            draw_circle(p.x, p.y, body.radius, body.color);
        }
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

    fn hud(&self) -> Vec<String> {
        vec![
            format!("year {:.2}   speed = {:.2} yr/s (UP/DOWN)", 2000.0 + self.t, self.speed),
            format!("energy drift = {:+.1e}", (self.energy() - self.energy0) / self.energy0.abs()),
            format!("|momentum| = {:.1e} Msun AU/yr", self.momentum().length()),
            "distances: sqrt(AU) from the Sun; Moon-Earth x100".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The system after `years` of simulated time using steps of `dt`.
    fn simulate(years: f64, dt: f64) -> Planets {
        let mut planets = Planets::new();
        let steps = (years / dt).floor();
        for _ in 0..steps as usize {
            planets.step(dt);
        }
        planets.step(years - steps * dt); // the remainder, so we end at exactly `years`
        planets
    }

    /// Positions after `years` of simulated time using steps of `dt`.
    fn positions_after(years: f64, dt: f64) -> Vec<DVec3> {
        simulate(years, dt).bodies.iter().map(|body| body.pos).collect()
    }

    #[test]
    fn conserves_energy_and_momentum() {
        let mut planets = Planets::new();
        assert!(planets.momentum().length() < 1e-14);

        // 20 years: ~1.7 Jupiter orbits, 83 Mercury orbits.
        for _ in 0..(20.0 / DT) as usize {
            planets.step(DT);
        }
        let drift = (planets.energy() - planets.energy0) / planets.energy0.abs();
        assert!(drift.abs() < 1e-12, "energy drift {drift}");
        assert!(planets.momentum().length() < 1e-14, "momentum {}", planets.momentum().length());
    }

    #[test]
    fn moon_stays_in_orbit_around_earth() {
        let mut planets = Planets::new();
        // Lunar distance is 0.0024-0.0027 AU in reality (perigee-apogee); allow some margin.
        for _ in 0..(5.0 / DT) as usize {
            planets.step(DT);
            let d = (planets.bodies[MOON].pos - planets.bodies[EARTH].pos).length();
            assert!(d > 0.0022 && d < 0.0029, "Moon-Earth distance {d} AU");
        }
    }

    #[test]
    fn integrator_is_sixth_order() {
        // Halving the step should shrink the error ~2^6 = 64 times. The error is measured against
        // a run with a much smaller step, for Venus and Mars. Smaller errors than ~1e-12 AU would
        // be rounding noise (as for Jupiter), so the step is large; Earth is left out because the
        // Moon's orbit isn't resolved at this step.
        let dt = 0.016;
        let reference = positions_after(4.0, dt / 8.0);
        let error = |dt: f64| {
            let pos = positions_after(4.0, dt);
            [2, 4].map(|i| (pos[i] - reference[i]).length())
        };
        let (coarse, fine) = (error(dt), error(dt / 2.0));
        for (c, f) in coarse.iter().zip(&fine) {
            let ratio = c / f;
            assert!(ratio > 40.0 && ratio < 90.0, "error ratio {ratio} (errors {c:e} and {f:e})");
        }
    }

    /// Positions of the planets and Moon relative to the Sun on 2050-01-01 12:00 TDB, from JPL Horizons.
    const HORIZONS_2050: [[f64; 3]; 9] = [
    [-0.19383887495353835, 0.2602374876728299, 0.03904203706874512], // Mercury
    [0.15161930466081194, -0.7113792666797559, -0.01856539064507374], // Venus
    [-0.18021419476575784, 0.9666841679689496, -0.00010945166455960886], // Earth
    [-1.5407874533325585, -0.509637674924416, 0.02701532942919058], // Mars
    [-2.3944545677412465, 4.6625244775693435, 0.034048340533462694], // Jupiter
    [4.768522500057428, -8.772394871483622, -0.03799131630296051], // Saturn
    [-17.823691404915362, 4.069472222215995, 0.24584878419787667], // Uranus
    [17.39691254047843, 24.19870912325753, -0.8992312613560624], // Neptune
    [-0.17790332795530533, 0.967745484807643, 1.973134179726551e-05], // Moon
    ];
    /// Same, on 2100-01-01 12:00 TDB.
    const HORIZONS_2100: [[f64; 3]; 9] = [
    [0.24751127983976062, -0.34788594901896175, -0.051118551807743585], // Mercury
    [0.6842480981278593, 0.2373629026651644, -0.036129345730427836], // Venus
    [-0.16603049156302818, 0.9692335221873367, -0.0002173730604088772], // Earth
    [0.603509860529159, 1.3841620865187276, 0.01434486121199327], // Mars
    [-5.373175933860913, -0.9120303811118975, 0.1238895958580123], // Jupiter
    [-9.152346961525364, -3.0816461612304074, 0.4182562731062589], // Saturn
    [18.864412332247717, 6.55067331750316, -0.2196427851129807], // Uranus
    [-29.059666113569605, 8.233668973110179, 0.5002350894643763], // Neptune
    [-0.16840341577891968, 0.969958343450389, -0.00019673126888694001], // Moon
    ];

    /// Error of each simulated body after `days` as an angle (radians): the distance from JPL's
    /// position divided by the distance to the Sun. The Moon is compared relative to Earth, since
    /// seen from the Sun it would mostly measure Earth's error.
    fn angle_errors_against(days: f64, expected: &[[f64; 3]; 9]) -> Vec<f64> {
        let pos = positions_after(days / DAYS_PER_YEAR, DT);
        let expected = expected.map(DVec3::from);
        (1..10)
            .map(|i| {
                let (error, distance) = if i == MOON {
                    (pos[MOON] - pos[EARTH] - (expected[MOON - 1] - expected[EARTH - 1]), (expected[MOON - 1] - expected[EARTH - 1]).length())
                } else {
                    (pos[i] - pos[0] - expected[i - 1], expected[i - 1].length())
                };
                error.length() / distance
            })
            .collect()
    }

    #[test]
    fn agrees_with_jpl_ephemeris() {
        // Planets within 0.05 arcseconds as seen from the Sun, the Moon within 25 arcseconds as seen
        // from Earth. What is left is mostly what the model leaves out: asteroids, Pluto, and the
        // Moon's shape and tides.
        let arcsec = std::f64::consts::PI / 180.0 / 3600.0;
        for (days, expected) in [(18_263.0, &HORIZONS_2050), (36_525.0, &HORIZONS_2100)] {
            for (i, error) in angle_errors_against(days, expected).iter().enumerate() {
                let limit = if i + 1 == MOON { 25.0 * arcsec } else { 0.05 * arcsec };
                assert!(*error < limit, "body {} off by {:.3} arcsec after {days} days", i + 1, error / arcsec);
            }
        }
    }

    #[test]
    fn moon_period_is_the_sidereal_month() {
        let mut planets = Planets::new();
        let angle = |planets: &Planets| {
            (planets.bodies[MOON].pos - planets.bodies[EARTH].pos).truncate().to_angle()
        };
        // Time the Moon takes to go around Earth: count passes through angle 0 going counter-clockwise.
        let (mut last, mut passes, mut first_t, mut last_t) = (angle(&planets), 0, 0.0, 0.0);
        for _ in 0..(20.0 / DT) as usize {
            planets.step(DT);
            let now = angle(&planets);
            if last < 0.0 && now >= 0.0 && last > -1.0 {
                passes += 1;
                if passes == 1 {
                    first_t = planets.t;
                }
                last_t = planets.t;
            }
            last = now;
        }
        let days = (last_t - first_t) / (passes - 1) as f64 * DAYS_PER_YEAR;
        assert!((days - 27.3217).abs() < 0.02, "period {days} days");
    }
}
