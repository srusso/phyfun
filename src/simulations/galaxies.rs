use std::f64::consts::PI;

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: length in kpc, time in Myr, mass in solar masses.
const KPC_KM: f64 = 3.085_677_581_491_367e16; // IAU 2015 Resolution B2
const MYR_S: f64 = 3.155_76e13; // 1e6 Julian years
const SUN_GM: f64 = 132_712_440_041.939_38; // km^3/s^2, as in planets.rs
/// G in kpc^3 / (Msun Myr^2).
const G: f64 = SUN_GM * MYR_S * MYR_S / (KPC_KM * KPC_KM * KPC_KM);
/// One kpc/Myr in km/s, for the HUD.
const KM_S: f64 = KPC_KM / MYR_S;

/// Both galaxies: total mass and Hernquist scale length, fitted to the Milky Way's measured
/// rotation curve (Eilers et al. 2019: 229.0 km/s at the Sun, falling 1.7 km/s per kpc).
const GALAXY_MASS: f64 = 4e11; // Msun
const HALO_SCALE: f64 = 8.0; // kpc
const CLOSEST_APPROACH: f64 = 25.0; // kpc
/// Zero orbital energy: the galaxies fall together from rest at infinity.
const ECCENTRICITY: f64 = 1.0;
const T_BEFORE: f64 = 500.0; // Myr from the start to closest approach
const T_END: f64 = 700.0; // Myr after closest approach
const DT: f64 = 1.0; // Myr

const DISK_STARS: usize = 5000;
const R_IN: f64 = 2.0; // kpc
const R_OUT: f64 = 20.0;
/// Toomre & Toomre's geometry for the Antennae.
const INCLINATION: f64 = 60.0 * PI / 180.0;
const ARGUMENT: f64 = -30.0 * PI / 180.0;
/// Stars farther than this from their galaxy count as pulled out of the disk.
const DISPLACED_RADIUS: f64 = 2.0 * R_OUT;

const START_SPEED: f64 = 60.0; // Myr per second of real time
const MIN_SPEED: f64 = 15.0;
const MAX_SPEED: f64 = 240.0;
const STAR_PIXELS: f32 = 1.5;
const CENTRE_PIXELS: f32 = 4.0;

/// Acceleration of a star at offset `d_vec` from the centre of a Hernquist sphere (Hernquist 1990)
/// of mass `m` and scale length `a`: potential -G m / (r + a). Bounded at the centre, so no
/// softening is needed. `a = 0` is a point mass.
fn hernquist_acceleration(m: f64, a: f64, d_vec: DVec3) -> DVec3 {
    let d = d_vec.length();
    d_vec * (-G * m / (d * (d + a) * (d + a)))
}

/// Circular speed at radius `r` in a Hernquist sphere, kpc/Myr.
fn circular_speed(m: f64, a: f64, r: f64) -> f64 {
    (G * m * r).sqrt() / (r + a)
}

/// Potential energy of two rigid Hernquist spheres with the same `a`, divided by -G M1 M2:
/// `w(d/a) / a`. Derived by averaging one sphere's potential over the other's density shell by
/// shell. For `a = 0` it is `1/d`.
fn pair_potential(a: f64, d: f64) -> f64 {
    if a == 0.0 {
        return 1.0 / d;
    }
    let x = d / a;
    let w = (x * x - 2.0) / (x * x * (x + 2.0)) + 4.0 * (x + 1.0) * x.ln_1p() / (x.powi(3) * (x + 2.0).powi(2));
    w / a
}

/// Attractive force between two rigid Hernquist spheres, divided by G M1 M2: `f(d/a) / a^2`
/// (`1/d^2` for `a = 0`). The terms cancel for small x, which is harmless here: the centres
/// never come closer than x = 3.
fn pair_force(a: f64, d: f64) -> f64 {
    if a == 0.0 {
        return 1.0 / (d * d);
    }
    let x = d / a;
    let f = (x.powi(3) - 6.0 * x - 8.0) / (x.powi(3) * (x + 2.0).powi(2))
        + 4.0 * ((4.0 * x * x + 9.0 * x + 6.0) * x.ln_1p() - x * (x + 2.0)) / (x.powi(4) * (x + 2.0).powi(3));
    f / (a * a)
}

/// The centre of a galaxy: one rigid, spherical mass distribution.
struct Galaxy {
    pos: DVec3,
    v: DVec3,
    a: DVec3,
    m: f64,
}

/// A massless tracer that feels both galaxies.
struct Star {
    pos: DVec3,
    v: DVec3,
    a: DVec3,
}

/// A disk of stars on circular orbits: inclination to the orbit plane (0 prograde, pi retrograde)
/// and argument of pericentre (angle in the orbit plane from the ascending node to the companion
/// at closest approach), radians.
struct Disk {
    stars: usize,
    inclination: f64,
    argument: f64,
}

struct Setup {
    m: [f64; 2],
    halo_scale: f64,
    r_p: f64,
    e: f64,
    t_before: f64,
    disks: [Option<Disk>; 2],
}

impl Setup {
    fn standard() -> Self {
        let disk = || Some(Disk { stars: DISK_STARS, inclination: INCLINATION, argument: ARGUMENT });
        Self { m: [GALAXY_MASS; 2], halo_scale: HALO_SCALE, r_p: CLOSEST_APPROACH, e: ECCENTRICITY, t_before: T_BEFORE, disks: [disk(), disk()] }
    }
}

#[derive(PartialEq)]
enum View {
    FaceOn,
    EdgeOn,
}

/// Two Milky-Way-like disk galaxies fall together from rest and swing past each other once at
/// 25 kpc; gravity alone pulls a bridge and tails out of their disks (Toomre & Toomre 1972, the
/// Mice and the Antennae). The restricted three-body problem in 3D: each galaxy is a rigid
/// Hernquist sphere fitted to the Milky Way's rotation curve, the two attract each other exactly,
/// and 5000 massless stars per disk feel both. A disk turning the way the orbit does is torn open,
/// one turning the other way survives. Leapfrog, 1 Myr steps. Time 0 is closest approach.
/// Not modelled: dynamical friction and halo deformation (the galaxies would fall back and merge;
/// the run stops at +700 Myr because of it), long tails, self-gravity of the disks (spiral arms,
/// clumps), a separate bulge, the halo beyond 25 kpc, and gas.
pub struct Galaxies {
    galaxies: [Galaxy; 2],
    stars: Vec<Star>,
    /// Which galaxy each star started in (stars are stored in blocks, galaxy 1 first).
    star_counts: [usize; 2],
    halo_scale: f64,
    /// Myr; 0 at closest approach.
    t: f64,
    /// Energy of the pair of centres at the start.
    energy0: f64,
    speed: f64,
    view: View,
}

impl Galaxies {
    pub fn new() -> Self {
        Self::encounter(&Setup::standard())
    }

    /// Builds the encounter: the centres are set up at closest approach in the centre-of-mass
    /// frame (orbit in the x-y plane, angular momentum along +z) and stepped backwards to the
    /// start, which is exact because leapfrog is time-reversible.
    fn encounter(setup: &Setup) -> Self {
        let [m1, m2] = setup.m;
        // Relative position and velocity of galaxy 2 with respect to galaxy 1 at closest approach.
        let v_p = ((1.0 + setup.e) * G * (m1 + m2) * pair_potential(setup.halo_scale, setup.r_p)).sqrt();
        let (rel_pos, rel_v) = (dvec3(0.0, setup.r_p, 0.0), dvec3(-v_p, 0.0, 0.0));
        let total = m1 + m2;
        let galaxy = |m: f64, share: f64| Galaxy { pos: rel_pos * share, v: rel_v * share, a: DVec3::ZERO, m };
        let mut sim = Self {
            galaxies: [galaxy(m1, -m2 / total), galaxy(m2, m1 / total)],
            stars: Vec::new(),
            star_counts: [0, 0],
            halo_scale: setup.halo_scale,
            t: 0.0,
            energy0: 0.0,
            speed: START_SPEED,
            view: View::FaceOn,
        };
        sim.update_accelerations();
        for _ in 0..(setup.t_before / DT).round() as usize {
            sim.leapfrog(-DT);
        }

        for (index, disk) in setup.disks.iter().enumerate() {
            if let Some(disk) = disk {
                sim.add_disk(index, disk);
            }
        }
        sim.update_accelerations();
        sim.energy0 = sim.pair_energy();
        sim
    }

    /// Adds `disk.stars` stars on exactly circular orbits around galaxy `index`, spread evenly in
    /// area between `R_IN` and `R_OUT` (deterministic: radius by area, azimuth by golden angle).
    fn add_disk(&mut self, index: usize, disk: &Disk) {
        let (sin_i, cos_i) = disk.inclination.sin_cos();
        let (sin_w, cos_w) = disk.argument.sin_cos();
        let spin = if index == 0 { dvec3(-sin_i * cos_w, sin_i * sin_w, cos_i) } else { dvec3(sin_i * cos_w, -sin_i * sin_w, cos_i) };
        let node = spin.cross(DVec3::Z);
        let e1 = if node.length() < 1e-9 { DVec3::X } else { node.normalize() };
        let e2 = spin.cross(e1);

        let centre = &self.galaxies[index];
        let (centre_pos, centre_v, m) = (centre.pos, centre.v, centre.m);
        for k in 0..disk.stars {
            let r = (R_IN * R_IN + (R_OUT * R_OUT - R_IN * R_IN) * (k as f64 + 0.5) / disk.stars as f64).sqrt();
            let phi = k as f64 * PI * (3.0 - 5f64.sqrt()); // golden angle
            let (sin, cos) = phi.sin_cos();
            self.stars.push(Star {
                pos: centre_pos + (e1 * cos + e2 * sin) * r,
                v: centre_v + (e2 * cos - e1 * sin) * circular_speed(m, self.halo_scale, r),
                a: DVec3::ZERO,
            });
        }
        self.star_counts[index] += disk.stars;
    }

    /// Sets the acceleration of the centres (from the exact pair force) and of every star.
    fn update_accelerations(&mut self) {
        let [g1, g2] = &mut self.galaxies;
        let separation = g2.pos - g1.pos;
        let d = separation.length();
        let u = separation / d;
        // Written as G M_other * force, never force / mass, so a massless companion works.
        let force = pair_force(self.halo_scale, d);
        g1.a = u * (G * g2.m * force);
        g2.a = -u * (G * g1.m * force);

        for star in &mut self.stars {
            star.a = hernquist_acceleration(g1.m, self.halo_scale, star.pos - g1.pos)
                + hernquist_acceleration(g2.m, self.halo_scale, star.pos - g2.pos);
        }
    }

    /// One kick-drift-kick leapfrog step for the centres and the stars: second order, symplectic,
    /// time-reversible, one force evaluation.
    fn leapfrog(&mut self, dt: f64) {
        for galaxy in &mut self.galaxies {
            galaxy.v += galaxy.a * (dt / 2.0);
            galaxy.pos += galaxy.v * dt;
        }
        for star in &mut self.stars {
            star.v += star.a * (dt / 2.0);
            star.pos += star.v * dt;
        }
        self.update_accelerations();
        for galaxy in &mut self.galaxies {
            galaxy.v += galaxy.a * (dt / 2.0);
        }
        for star in &mut self.stars {
            star.v += star.a * (dt / 2.0);
        }
        self.t += dt;
    }

    fn kinetic_energy(&self) -> f64 {
        self.galaxies.iter().map(|galaxy| 0.5 * galaxy.m * galaxy.v.length_squared()).sum()
    }

    /// Kinetic energy of the centres plus the exact potential energy of the two spheres.
    fn pair_energy(&self) -> f64 {
        let [g1, g2] = &self.galaxies;
        self.kinetic_energy() - G * g1.m * g2.m * pair_potential(self.halo_scale, self.separation())
    }

    #[cfg(test)]
    fn pair_angular_momentum(&self) -> DVec3 {
        self.galaxies.iter().map(|galaxy| galaxy.pos.cross(galaxy.v) * galaxy.m).sum()
    }

    fn separation(&self) -> f64 {
        (self.galaxies[1].pos - self.galaxies[0].pos).length()
    }

    /// Share of the stars that started in galaxy `home` and are now farther than 40 kpc from it.
    fn displaced_fraction(&self, home: usize) -> f64 {
        let start = if home == 0 { 0 } else { self.star_counts[0] };
        let stars = &self.stars[start..start + self.star_counts[home]];
        let centre = self.galaxies[home].pos;
        stars.iter().filter(|star| (star.pos - centre).length() > DISPLACED_RADIUS).count() as f64 / stars.len().max(1) as f64
    }
}

impl Simulation for Galaxies {
    fn step(&mut self, dt: f64) {
        self.leapfrog(dt);
    }

    fn is_over(&self) -> bool {
        self.t >= T_END
    }

    fn dt(&self) -> f64 {
        DT
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn world_bounds(&self) -> Rect {
        // Centred on the centre of mass; every star stays inside to +1000 Myr.
        Rect::new(-150.0, -100.0, 300.0, 200.0)
    }

    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
        if is_key_pressed(KeyCode::V) {
            self.view = if self.view == View::FaceOn { View::EdgeOn } else { View::FaceOn };
        }
    }

    fn draw(&self) {
        let bounds = self.world_bounds();
        let pixel = (bounds.w / screen_width()).max(bounds.h / screen_height());
        // Face-on shows (x, y), edge-on (x, z).
        let project = |pos: DVec3| match self.view {
            View::FaceOn => vec2(pos.x as f32, pos.y as f32),
            View::EdgeOn => vec2(pos.x as f32, pos.z as f32),
        };
        let colors = [Color::new(0.53, 0.81, 0.92, 0.6), Color::new(1.0, 0.65, 0.0, 0.6)]; // SKYBLUE, ORANGE
        let side = STAR_PIXELS * pixel;
        let mut first = 0;
        for (home, count) in self.star_counts.iter().enumerate() {
            for star in &self.stars[first..first + count] {
                let p = project(star.pos);
                draw_rectangle(p.x - side / 2.0, p.y - side / 2.0, side, side, colors[home]);
            }
            first += count;
        }
        let side = CENTRE_PIXELS * pixel;
        for galaxy in &self.galaxies {
            let p = project(galaxy.pos);
            draw_rectangle(p.x - side / 2.0, p.y - side / 2.0, side, side, WHITE);
        }
    }

    fn hud(&self) -> Vec<String> {
        let relative_speed = (self.galaxies[1].v - self.galaxies[0].v).length() * KM_S;
        let kinetic = self.kinetic_energy();
        vec![
            format!("t = {:+.0} Myr from closest approach   speed = {} Myr/s (UP/DOWN)", self.t, self.speed),
            format!("separation = {:.1} kpc   relative speed = {:.0} km/s", self.separation(), relative_speed),
            format!(
                "stars pulled beyond 40 kpc of their galaxy: {:.0}% / {:.0}%",
                100.0 * self.displaced_fraction(0),
                100.0 * self.displaced_fraction(1)
            ),
            format!(
                "orbit energy error = {:.1e}   view: {} (V)",
                ((self.pair_energy() - self.energy0) / kinetic).abs(),
                if self.view == View::FaceOn { "face-on" } else { "edge-on" }
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(sim: &mut Galaxies, steps: usize) {
        for _ in 0..steps {
            sim.leapfrog(DT);
        }
    }

    fn setup_without_disks() -> Setup {
        Setup { disks: [None, None], ..Setup::standard() }
    }

    fn setup_with_stars(stars: usize, inclinations: [f64; 2]) -> Setup {
        let disk = |inclination| Some(Disk { stars, inclination, argument: ARGUMENT });
        Setup { disks: [disk(inclinations[0]), disk(inclinations[1])], ..Setup::standard() }
    }

    fn close(value: f64, expected: f64, relative: f64, what: &str) {
        assert!((value - expected).abs() <= relative * expected.abs(), "{what}: {value}, expected {expected}");
    }

    #[test]
    fn units() {
        close(G, 4.498_502e-12, 1e-6, "G");
        close(KM_S, 977.792_2, 1e-6, "km/s per kpc/Myr");
    }

    #[test]
    fn rotation_curve_matches_the_milky_way() {
        for r in [5.0, 8.122, 10.0, 15.0, 20.0, 25.0] {
            let model = circular_speed(GALAXY_MASS, HALO_SCALE, r) * KM_S;
            let measured = 229.0 - 1.7 * (r - 8.122);
            assert!((model / measured - 1.0).abs() < 0.04, "R = {r}: model {model}, measured {measured}");
        }

        // A fresh disk: each star moves at v_c(R), perpendicular to its radius and to the spin.
        let sim = Galaxies::encounter(&Setup { t_before: 0.0, ..setup_with_stars(300, [INCLINATION; 2]) });
        let centre = &sim.galaxies[0];
        for star in &sim.stars[..300] {
            let (rel, v) = (star.pos - centre.pos, star.v - centre.v);
            let r = rel.length();
            close(v.length(), circular_speed(GALAXY_MASS, HALO_SCALE, r), 1e-12, "speed");
            assert!(rel.dot(v).abs() < 1e-12 * r * v.length(), "velocity not perpendicular to radius");
        }
    }

    #[test]
    fn pair_force_is_exact() {
        for (x, f, w) in [(0.5, 0.059_028_45, f64::NAN), (1.0, 0.062_192_06, 0.282_797), (2.0, 0.046_658_17, 0.227_995), (5.0, 0.018_599_29, 0.138_449), (10.0, 0.006_719_758, 0.082_399_4)] {
            close(pair_force(1.0, x), f, 1e-6, "f");
            if !w.is_nan() {
                // The reference values of w are rounded to six digits.
                assert!((pair_potential(1.0, x) - w).abs() < 1e-6, "w({x}) = {}, expected {w}", pair_potential(1.0, x));
            }
        }
        close(pair_force(1.0, 20.0), 0.002_044_791, 1e-6, "f(20)");

        // The force is minus the slope of the potential energy.
        for d in [0.5, 2.0, 10.0] {
            let h = 1e-5 * d;
            let slope = (pair_potential(1.0, d + h) - pair_potential(1.0, d - h)) / (2.0 * h);
            close(pair_force(1.0, d), -slope, 1e-7, "force against potential");
        }

        // Against the definition: the pull of one sphere on the other, summed shell by shell
        // (midpoint rule in the enclosed mass fraction m and the angle cosine mu).
        let n = 400;
        for d in [0.5, 2.0, 5.0] {
            let mut sum = 0.0;
            for i in 0..n {
                let m = (i as f64 + 0.5) / n as f64;
                let r = m.sqrt() / (1.0 - m.sqrt());
                for j in 0..n {
                    let mu = -1.0 + 2.0 * (j as f64 + 0.5) / n as f64;
                    let s = (r * r + d * d - 2.0 * r * d * mu).sqrt();
                    sum += (d - r * mu) / (s * (s + 1.0) * (s + 1.0));
                }
            }
            close(sum / (n * n) as f64, pair_force(1.0, d), 5e-5, "quadrature");
        }
    }

    #[test]
    fn pair_follows_a_kepler_orbit() {
        // Point masses on TT72's Antennae orbit: semi-major axis 50 kpc.
        let setup = Setup { m: [1e11; 2], halo_scale: 0.0, e: 0.5, t_before: 0.0, ..setup_without_disks() };
        let mut sim = Galaxies::encounter(&setup);
        let start = sim.galaxies[0].pos;
        let mut worst: f64 = 0.0;
        for step in 1..=2342 {
            sim.leapfrog(DT);
            worst = worst.max(((sim.pair_energy() - sim.energy0) / sim.energy0).abs());
            if step == 1171 {
                assert!((sim.separation() - 75.0).abs() < 0.01, "separation at half period {}", sim.separation());
            }
        }
        assert!((sim.separation() - 25.0).abs() < 1e-4, "separation after a period {}", sim.separation());
        assert!((sim.galaxies[0].pos - start).length() < 0.03, "galaxy 1 off by {}", (sim.galaxies[0].pos - start).length());
        assert!(worst < 1e-4, "energy error {worst}");
    }

    #[test]
    fn pair_conserves_energy_and_angular_momentum() {
        let mut sim = Galaxies::encounter(&setup_without_disks());
        assert!((sim.separation() - 134.6).abs() < 0.1, "starts {} kpc apart", sim.separation());
        let (mut min_sep, mut min_t, mut energy, mut momentum): (f64, f64, f64, f64) = (f64::INFINITY, 0.0, 0.0, 0.0);
        let l0 = sim.pair_angular_momentum();
        let mut relative_speed_at_zero = 0.0;
        for _ in 0..(T_BEFORE + T_END) as usize {
            if sim.separation() < min_sep {
                (min_sep, min_t) = (sim.separation(), sim.t);
            }
            if sim.t.abs() < 1e-9 {
                relative_speed_at_zero = (sim.galaxies[1].v - sim.galaxies[0].v).length() * KM_S;
            }
            sim.leapfrog(DT);
            energy = energy.max(((sim.pair_energy() - sim.energy0) / sim.kinetic_energy()).abs());
            momentum = momentum.max((sim.pair_angular_momentum() - l0).length() / l0.length());
            let total_p = (sim.galaxies[0].v * sim.galaxies[0].m + sim.galaxies[1].v * sim.galaxies[1].m).length();
            assert!(total_p < 1e-12 * GALAXY_MASS * 398.2 / KM_S, "total momentum {total_p}");
        }
        assert!((min_sep - 25.0).abs() < 1e-6 && min_t.abs() < 1e-9, "closest approach {min_sep} kpc at t = {min_t}");
        assert!((relative_speed_at_zero - 398.2).abs() < 0.1, "relative speed {relative_speed_at_zero} km/s");
        assert!(energy < 5e-5, "energy error {energy}");
        assert!(momentum < 1e-12, "angular momentum change {momentum}");
    }

    #[test]
    fn isolated_disk_keeps_its_shape() {
        let setup = Setup {
            m: [GALAXY_MASS, 0.0],
            t_before: 0.0,
            disks: [Some(Disk { stars: 500, inclination: 0.0, argument: 0.0 }), None],
            ..Setup::standard()
        };
        let mut sim = Galaxies::encounter(&setup);
        let start: Vec<DVec3> = sim.stars.iter().map(|star| star.pos - sim.galaxies[0].pos).collect();
        run(&mut sim, 2700);
        for (k, star) in sim.stars.iter().enumerate() {
            let rel = star.pos - sim.galaxies[0].pos;
            let r0 = start[k].length();
            assert!((rel.length() / r0 - 1.0).abs() < 0.005, "star {k} moved from {r0} to {}", rel.length());
            let expected = start[k].y.atan2(start[k].x) + circular_speed(GALAXY_MASS, HALO_SCALE, r0) / r0 * sim.t;
            let off = (rel.y.atan2(rel.x) - expected + PI).rem_euclid(2.0 * PI) - PI;
            assert!(off.abs() < 0.15, "star {k} is {off} rad off its azimuth");
        }
    }

    #[test]
    fn stars_conserve_the_jacobi_integral() {
        // Two galaxies on a circular orbit: a star's energy in the rotating frame is conserved.
        let r_p = 60.0;
        let e = pair_force(HALO_SCALE, r_p) * r_p / pair_potential(HALO_SCALE, r_p) - 1.0;
        let setup = Setup {
            r_p,
            e,
            t_before: 0.0,
            disks: [Some(Disk { stars: 500, inclination: INCLINATION, argument: ARGUMENT }), None],
            ..Setup::standard()
        };
        let mut sim = Galaxies::encounter(&setup);
        let omega = (sim.galaxies[1].v - sim.galaxies[0].v).length() / sim.separation();
        let jacobi = |sim: &Galaxies, star: &Star| {
            let [g1, g2] = &sim.galaxies;
            let (d1, d2) = ((star.pos - g1.pos).length(), (star.pos - g2.pos).length());
            0.5 * star.v.length_squared()
                - G * g1.m / (d1 + HALO_SCALE)
                - G * g2.m / (d2 + HALO_SCALE)
                - omega * (star.pos.x * star.v.y - star.pos.y * star.v.x)
        };
        let start: Vec<f64> = sim.stars.iter().map(|star| jacobi(&sim, star)).collect();
        let scale = circular_speed(GALAXY_MASS, HALO_SCALE, R_OUT).powi(2);
        let mut separation_error: f64 = 0.0;
        let mut worst: f64 = 0.0;
        for _ in 0..1000 {
            sim.leapfrog(DT);
            separation_error = separation_error.max((sim.separation() - 60.0).abs());
            for (star, c0) in sim.stars.iter().zip(&start) {
                worst = worst.max((jacobi(&sim, star) - c0).abs() / scale);
            }
        }
        assert!(separation_error < 1e-3, "separation changed by {separation_error}");
        assert!(worst < 2e-3, "Jacobi integral changed by {worst} of v_c^2");
    }

    #[test]
    fn is_time_reversible() {
        let mut sim = Galaxies::encounter(&setup_with_stars(200, [INCLINATION; 2]));
        let start: Vec<DVec3> = sim.stars.iter().map(|star| star.pos).collect();
        let centres = [sim.galaxies[0].pos, sim.galaxies[1].pos];
        run(&mut sim, 800);
        for _ in 0..800 {
            sim.leapfrog(-DT);
        }
        for (star, p0) in sim.stars.iter().zip(&start) {
            assert!((star.pos - *p0).length() < 1e-9, "star off by {}", (star.pos - *p0).length());
        }
        for (galaxy, p0) in sim.galaxies.iter().zip(centres) {
            assert!((galaxy.pos - p0).length() < 1e-9);
        }
    }

    #[test]
    fn prograde_disk_is_torn_retrograde_survives() {
        let mut sim = Galaxies::encounter(&setup_with_stars(1000, [0.0, PI]));
        run(&mut sim, (T_BEFORE + 400.0) as usize);
        let (prograde, retrograde) = (sim.displaced_fraction(0), sim.displaced_fraction(1));
        assert!(prograde > 0.30, "prograde disk lost {prograde}");
        assert!(retrograde < 0.02, "retrograde disk lost {retrograde}");
    }

    #[test]
    fn identical_galaxies_stay_mirror_images() {
        let mut sim = Galaxies::encounter(&setup_with_stars(2000, [INCLINATION; 2]));
        let check = |sim: &Galaxies| {
            for k in 0..2000 {
                let (a, b) = (sim.stars[k].pos, sim.stars[2000 + k].pos);
                assert!((b - dvec3(-a.x, -a.y, a.z)).length() < 1e-9, "star {k} is not the mirror image");
            }
        };
        check(&sim);
        run(&mut sim, (T_BEFORE + T_END) as usize);
        check(&sim);
        for home in 0..2 {
            let fraction = sim.displaced_fraction(home);
            assert!(fraction > 0.22 && fraction < 0.30, "galaxy {home} lost {fraction}");
        }
    }
}
