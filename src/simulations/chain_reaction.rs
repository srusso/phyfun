use std::collections::VecDeque;
use std::f64::consts::PI;

use macroquad::prelude::*;
use macroquad::rand::RandGenerator;

use crate::simulation::Simulation;

// Units: length in cm, time in ns, microscopic cross sections in barns (1 b = 1e-24 cm^2),
// macroscopic cross sections (Sigma = n sigma) in 1/cm, mass in kg for display only.

const AVOGADRO: f64 = 6.022_140_76e23; // 1/mol, exact (SI definition)
const U235_MOLAR_MASS: f64 = 235.0439; // g/mol
/// Godiva's density, so radii compare directly with that real bare sphere.
const DENSITY: f64 = 18.74; // g/cm^3

// One-group cross sections of U-235 averaged over the fission spectrum, JENDL-4.0
// (https://wwwndc.jaea.go.jp/cgi-bin/Tab80WWW.cgi?lib=J40&iso=U235).
const SIGMA_FISSION: f64 = 1.218; // b
const SIGMA_CAPTURE: f64 = 0.08741; // b
const SIGMA_ELASTIC: f64 = 4.451; // b
const SIGMA_INELASTIC: f64 = 1.858; // b
/// (n,2n) is counted as plain scattering: the second neutron is dropped.
const SIGMA_N2N: f64 = 0.01243; // b
/// Neutrons per fission. Reed, "The Physics of the Manhattan Project", table 2.1 (quoted from memory).
const NU: f64 = 2.637;
/// Speed of a 1 MeV neutron, the energy whose speed matches the fission-spectrum average of 1/v:
/// c sqrt(2E / m c^2) with m c^2 = 939.565 MeV.
const V: f64 = 1.3832; // cm/ns

/// Critical radii to compare with: one-group transport theory for this data (exact solution of
/// the integral transport equation), and Godiva, a real bare sphere of 93.71% enriched uranium.
const THEORY_RADIUS: f64 = 7.486; // cm
const GODIVA_RADIUS: f64 = 8.7407; // cm

const START_RADIUS: f64 = 6.5; // cm
const MIN_RADIUS: f64 = 3.0;
const MAX_RADIUS: f64 = 12.0;
const N_TARGET: usize = 2000;
const SEED: u64 = 7;
/// Tallies start this long after the radius was last set (six lifetimes), so the spatial shape
/// of the neutron population has converged.
const SETTLE: f64 = 30.0; // ns

const START_SPEED: f64 = 5.0; // simulated ns per second of real time
const MIN_SPEED: f64 = 1.25;
const MAX_SPEED: f64 = 160.0;

// Drawing.
const NEUTRON_SIZE: f32 = 0.24; // cm
const FLASH_LIFE: f64 = 0.15; // s of real time
const MAX_FLASHES: usize = 200;
/// Neutrons that left the sphere are drawn until they are this far outside it.
const ESCAPE_FADE: f64 = 6.0; // cm
const MAX_ESCAPED: usize = 5000;

/// Macroscopic cross sections (1/cm) and neutrons per fission.
struct Material {
    sigma_f: f64,
    sigma_c: f64,
    sigma_s: f64,
    nu: f64,
}

impl Material {
    fn u235() -> Self {
        // Atom density in atoms/(b cm).
        let n = DENSITY * AVOGADRO / U235_MOLAR_MASS * 1e-24;
        Self {
            sigma_f: n * SIGMA_FISSION,
            sigma_c: n * SIGMA_CAPTURE,
            sigma_s: n * (SIGMA_ELASTIC + SIGMA_INELASTIC + SIGMA_N2N),
            nu: NU,
        }
    }

    fn sigma_t(&self) -> f64 {
        self.sigma_f + self.sigma_c + self.sigma_s
    }
}

/// Mass in kg of a sphere of U-235 metal.
fn sphere_mass(radius: f64) -> f64 {
    4.0 / 3.0 * PI * radius.powi(3) * DENSITY / 1000.0
}

#[derive(Clone, Copy)]
struct Neutron {
    pos: DVec3,
    dir: DVec3,
}

/// What happened to the neutrons since the tally was last reset.
#[derive(Default, PartialEq, Debug)]
struct Tally {
    fissions: u64,
    captures: u64,
    escapes: u64,
    /// Total time neutrons spent alive, ns.
    neutron_time: f64,
}

impl Tally {
    fn histories(&self) -> u64 {
        self.fissions + self.captures + self.escapes
    }

    /// Neutrons produced per neutron lost, in the population as it actually evolves.
    /// It satisfies `k - 1 = alpha * lifetime`.
    fn k(&self, nu: f64) -> f64 {
        nu * self.fissions as f64 / self.histories().max(1) as f64
    }

    fn k_error(&self, nu: f64) -> f64 {
        let n = self.histories().max(1) as f64;
        let p = self.fissions as f64 / n;
        nu * (p * (1.0 - p) / n).sqrt()
    }

    /// Mean lifetime of a neutron, ns.
    fn lifetime(&self) -> f64 {
        self.neutron_time / self.histories().max(1) as f64
    }

    /// Growth rate of the population, 1/ns.
    fn alpha(&self, nu: f64) -> f64 {
        (self.k(nu) - 1.0) / self.lifetime()
    }
}

fn uniform(rng: &RandGenerator) -> f64 {
    rng.gen_range(0.0_f64, 1.0)
}

/// Distance to the next collision: exponential with mean 1/sigma_t.
fn free_path(rng: &RandGenerator, sigma_t: f64) -> f64 {
    -(1.0 - uniform(rng)).ln() / sigma_t
}

fn isotropic_direction(rng: &RandGenerator) -> DVec3 {
    let mu = 2.0 * uniform(rng) - 1.0;
    let phi = 2.0 * PI * uniform(rng);
    let s = (1.0 - mu * mu).sqrt();
    dvec3(s * phi.cos(), s * phi.sin(), mu)
}

/// 2 or 3 with mean exactly `nu`.
fn neutrons_per_fission(rng: &RandGenerator, nu: f64) -> usize {
    (nu + uniform(rng)).floor() as usize
}

fn point_in_sphere(rng: &RandGenerator, radius: f64) -> DVec3 {
    isotropic_direction(rng) * (radius * uniform(rng).cbrt())
}

/// Neutrons flying through a bare sphere of uranium-235 metal (analog Monte Carlo, one energy
/// group): each travels an exponentially distributed distance, then scatters isotropically, is
/// captured, or splits the nucleus into 2 or 3 new neutrons; those that reach the surface are lost.
/// LEFT/RIGHT change the radius. Below the critical radius the population decays, above it it
/// grows, and `k` on the HUD crosses 1. The count is held near 2000 by cloning or dropping
/// neutrons (the HUD shows the true growth). Drawn projected along z.
/// Not modelled: neutron energy (one group costs ~14% in critical radius against Godiva),
/// forward-peaked scattering, delayed neutrons, Godiva's U-238, (n,2n), heating and expansion,
/// and chains dying out.
pub struct ChainReaction {
    material: Material,
    radius: f64,
    neutrons: Vec<Neutron>,
    /// Neutrons that left the sphere, kept only to be drawn flying off.
    escaped: Vec<Neutron>,
    /// Recent fissions: position and simulated time.
    flashes: VecDeque<(DVec3, f64)>,
    /// Population control scales the count by this much; the true population is
    /// `count * exp(log_weight)`.
    log_weight: f64,
    count_at_reset: usize,
    target: usize,
    tally: Tally,
    t: f64,
    /// Time the radius was last set, and how long to wait before tallying.
    t_reset: f64,
    settle: f64,
    t_next_control: f64,
    speed: f64,
    rng: RandGenerator,
}

impl ChainReaction {
    pub fn new() -> Self {
        Self::with(Material::u235(), START_RADIUS, N_TARGET, SEED)
    }

    fn with(material: Material, radius: f64, n_target: usize, seed: u64) -> Self {
        let rng = RandGenerator::new();
        rng.srand(seed);
        let neutrons = (0..n_target)
            .map(|_| Neutron { pos: point_in_sphere(&rng, radius), dir: isotropic_direction(&rng) })
            .collect();
        Self {
            material,
            radius,
            neutrons,
            escaped: Vec::new(),
            flashes: VecDeque::new(),
            log_weight: 0.0,
            count_at_reset: n_target,
            target: n_target,
            tally: Tally::default(),
            t: 0.0,
            t_reset: 0.0,
            settle: SETTLE,
            t_next_control: 1.0,
            speed: START_SPEED,
            rng,
        }
    }

    /// True population relative to when the radius was last set.
    fn relative_population(&self) -> f64 {
        self.neutrons.len() as f64 * self.log_weight.exp() / self.count_at_reset as f64
    }

    /// Keeps the number of simulated neutrons within a factor 2 of the target by dropping random
    /// neutrons or cloning random ones, and remembers the factor in `log_weight`.
    fn population_control(&mut self) {
        let count = self.neutrons.len();
        if count > 2 * self.target {
            while self.neutrons.len() > self.target {
                let i = self.rng.gen_range(0, self.neutrons.len());
                self.neutrons.swap_remove(i);
            }
        } else if count > 0 && count < self.target / 2 {
            while self.neutrons.len() < self.target {
                let i = self.rng.gen_range(0, count);
                self.neutrons.push(self.neutrons[i]);
            }
        } else {
            return;
        }
        self.log_weight += (count as f64 / self.target as f64).ln();
    }

    /// Sets a new radius: neutrons now outside fly off, and the tallies start again.
    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
        let (inside, outside): (Vec<Neutron>, Vec<Neutron>) =
            self.neutrons.iter().copied().partition(|neutron| neutron.pos.length() <= radius);
        self.neutrons = inside;
        self.escaped.extend(outside.into_iter().take(MAX_ESCAPED.saturating_sub(self.escaped.len())));
        self.tally = Tally::default();
        self.log_weight = 0.0;
        self.count_at_reset = self.neutrons.len().max(1);
        self.t_reset = self.t;
    }

    fn settled(&self) -> bool {
        self.t >= self.t_reset + self.settle
    }
}

impl Simulation for ChainReaction {
    fn step(&mut self, dt: f64) {
        let tallying = self.settled();
        let sigma_t = self.material.sigma_t();
        let radius = self.radius;

        // Escaped neutrons fly on in straight lines until they are far outside.
        for neutron in &mut self.escaped {
            neutron.pos += neutron.dir * (V * dt);
        }
        self.escaped.retain(|neutron| neutron.pos.length() < radius + ESCAPE_FADE);

        // Each neutron is followed for the whole step. A neutron born in a fission continues for
        // the rest of the step (`time_left`), so no generation is delayed. The free path is
        // memoryless, so cutting a flight at the end of a step changes nothing statistically.
        let mut stack: Vec<(Neutron, f64)> = self.neutrons.drain(..).map(|neutron| (neutron, dt)).collect();
        let mut alive = Vec::with_capacity(stack.len() + stack.len() / 4);
        while let Some((mut neutron, mut time_left)) = stack.pop() {
            loop {
                let to_collision = free_path(&self.rng, sigma_t);
                let b = neutron.pos.dot(neutron.dir);
                // Distance along the direction of flight to the surface, from inside.
                let to_surface = -b + (b * b + radius * radius - neutron.pos.length_squared()).max(0.0).sqrt();
                let reach = V * time_left;
                if reach < to_collision && reach < to_surface {
                    neutron.pos += neutron.dir * reach;
                    if tallying {
                        self.tally.neutron_time += time_left;
                    }
                    alive.push(neutron);
                    break;
                }
                if to_surface <= to_collision {
                    if tallying {
                        self.tally.escapes += 1;
                        self.tally.neutron_time += to_surface / V;
                    }
                    if self.escaped.len() < MAX_ESCAPED {
                        self.escaped.push(Neutron { pos: neutron.pos + neutron.dir * to_surface, dir: neutron.dir });
                    }
                    break;
                }
                neutron.pos += neutron.dir * to_collision;
                time_left -= to_collision / V;
                if tallying {
                    self.tally.neutron_time += to_collision / V;
                }

                // Which reaction: in proportion to the cross sections.
                let x = uniform(&self.rng) * sigma_t;
                if x < self.material.sigma_f {
                    if tallying {
                        self.tally.fissions += 1;
                    }
                    for _ in 0..neutrons_per_fission(&self.rng, self.material.nu) {
                        let child = Neutron { pos: neutron.pos, dir: isotropic_direction(&self.rng) };
                        stack.push((child, time_left));
                    }
                    if self.flashes.len() == MAX_FLASHES {
                        self.flashes.pop_front();
                    }
                    self.flashes.push_back((neutron.pos, self.t + dt - time_left));
                    break;
                } else if x < self.material.sigma_f + self.material.sigma_c {
                    if tallying {
                        self.tally.captures += 1;
                    }
                    break;
                }
                neutron.dir = isotropic_direction(&self.rng); // scattering
            }
        }
        self.neutrons = alive;

        self.t += dt;
        while self.flashes.front().is_some_and(|&(_, born)| self.t - born > FLASH_LIFE * self.speed) {
            self.flashes.pop_front();
        }
        // Once per simulated nanosecond, not every step: renormalising at the exact moment the
        // count crosses a bound selects unlucky configurations and biases the growth rate.
        if self.t >= self.t_next_control {
            self.population_control();
            self.t_next_control = self.t.floor() + 1.0;
        }
    }

    fn is_over(&self) -> bool {
        self.neutrons.is_empty()
    }

    fn dt(&self) -> f64 {
        self.speed / 240.0
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn world_bounds(&self) -> Rect {
        // The sphere is centred on the origin, to the right of the HUD text.
        Rect::new(-30.0, -17.0, 48.0, 32.0)
    }

    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
        let step = if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) { 1.0 } else { 0.1 };
        let mut radius = self.radius;
        if is_key_pressed(KeyCode::Right) {
            radius += step;
        }
        if is_key_pressed(KeyCode::Left) {
            radius -= step;
        }
        let radius = radius.clamp(MIN_RADIUS, MAX_RADIUS);
        if radius != self.radius {
            self.set_radius(radius);
        }
    }

    fn draw(&self) {
        // Projected along z: the sphere is a disc.
        let r = self.radius as f32;
        draw_poly(0.0, 0.0, 96, r, 0.0, Color::new(0.15, 0.15, 0.17, 1.0));
        draw_poly_lines(0.0, 0.0, 96, r, 0.0, 0.1, Color::new(0.5, 0.5, 0.55, 1.0));
        draw_circle_lines(0.0, 0.0, THEORY_RADIUS as f32, 0.08, GREEN);
        draw_circle_lines(0.0, 0.0, GODIVA_RADIUS as f32, 0.08, GRAY);

        for neutron in &self.escaped {
            let outside = (neutron.pos.length() - self.radius) / ESCAPE_FADE;
            let alpha = (1.0 - outside).clamp(0.0, 1.0) as f32;
            draw_rectangle(neutron.pos.x as f32 - NEUTRON_SIZE / 2.0, neutron.pos.y as f32 - NEUTRON_SIZE / 2.0, NEUTRON_SIZE, NEUTRON_SIZE, Color::new(0.3, 0.5, 1.0, alpha));
        }
        for neutron in &self.neutrons {
            // Nearer to the viewer (larger z) is brighter.
            let brightness = 0.45 + 0.55 * ((neutron.pos.z / self.radius + 1.0) / 2.0) as f32;
            draw_rectangle(neutron.pos.x as f32 - NEUTRON_SIZE / 2.0, neutron.pos.y as f32 - NEUTRON_SIZE / 2.0, NEUTRON_SIZE, NEUTRON_SIZE, Color::new(brightness, brightness, brightness, 1.0));
        }
        for &(pos, born) in &self.flashes {
            let age = ((self.t - born) / (FLASH_LIFE * self.speed)).clamp(0.0, 1.0) as f32;
            draw_circle_lines(pos.x as f32, pos.y as f32, 0.2 + 0.6 * age, 0.05, Color::new(1.0, 0.9, 0.2, 1.0 - age));
        }
    }

    fn hud(&self) -> Vec<String> {
        let nu = self.material.nu;
        let settled = self.settled() && self.tally.histories() > 0;
        let tally = &self.tally;
        let mut lines = vec![format!("radius {:.2} cm   mass {:.1} kg   (LEFT/RIGHT)", self.radius, sphere_mass(self.radius))];
        if settled {
            let alpha = tally.alpha(nu);
            lines.push(format!("k = {:.3} +- {:.3}", tally.k(nu), tally.k_error(nu)));
            lines.push(if alpha >= 0.0 {
                format!("growth {:+.4} /ns   doubles every {:.1} ns", alpha, 2f64.ln() / alpha)
            } else {
                format!("decay {:+.4} /ns   halves every {:.1} ns", alpha, 2f64.ln() / -alpha)
            });
        } else {
            lines.push("k: settling...".to_string());
            lines.push("growth: settling...".to_string());
        }
        lines.push(format!("population x {:.1e} since radius set", self.relative_population()));
        lines.push(format!("t = {:.1} ns   speed {} ns/s (UP/DOWN)", self.t, self.speed));
        if settled {
            let n = tally.histories() as f64 / 100.0;
            lines.push(format!(
                "fate: {:.0}% fission  {:.0}% capture  {:.0}% escape",
                tally.fissions as f64 / n,
                tally.captures as f64 / n,
                tally.escapes as f64 / n
            ));
        } else {
            lines.push("fate: settling...".to_string());
        }
        lines.push(format!("green: one-group theory, {:.2} cm, {:.1} kg", THEORY_RADIUS, sphere_mass(THEORY_RADIUS)));
        lines.push(format!("grey: Godiva (93.7% U-235), {:.2} cm, {:.1} kg", GODIVA_RADIUS, sphere_mass(GODIVA_RADIUS)));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settles for `settle` ns, zeroes the tally, then runs `tally` ns more in steps of 1 ns.
    fn run(material: Material, radius: f64, seed: u64, settle: usize, tally: usize) -> ChainReaction {
        let mut sim = ChainReaction::with(material, radius, N_TARGET, seed);
        sim.settle = 0.0;
        for _ in 0..settle {
            sim.step(1.0);
        }
        sim.tally = Tally::default();
        for _ in 0..tally {
            sim.step(1.0);
        }
        sim
    }

    /// Sood et al.'s one-group U-235 bare sphere (LA-13511), critical at 7.428998 cm.
    fn sood() -> Material {
        Material { sigma_f: 0.065280, sigma_c: 0.013056, sigma_s: 0.248064, nu: 2.70 }
    }

    fn k_at(radius: f64, seed: u64, tally: usize) -> f64 {
        let sim = run(Material::u235(), radius, seed, 50, tally);
        sim.tally.k(sim.material.nu)
    }

    #[test]
    fn constants_follow_from_the_table() {
        let m = Material::u235();
        let n = DENSITY * AVOGADRO / U235_MOLAR_MASS * 1e-24;
        assert!((n - 0.0480144).abs() < 1e-6, "n {n}");
        assert!((m.sigma_t() - 0.3661981).abs() < 1e-6, "sigma_t {}", m.sigma_t());
        let k_inf = m.nu * m.sigma_f / (m.sigma_f + m.sigma_c);
        assert!((k_inf - 2.4604).abs() < 1e-3, "k_inf {k_inf}");
        assert!((sphere_mass(GODIVA_RADIUS) - 52.42).abs() < 0.01);
    }

    #[test]
    fn free_paths_average_one_mean_free_path() {
        let rng = RandGenerator::new();
        rng.srand(1);
        let sigma_t = Material::u235().sigma_t();
        let mean = (0..200_000).map(|_| free_path(&rng, sigma_t)).sum::<f64>() / 200_000.0;
        assert!((mean * sigma_t - 1.0).abs() < 0.01, "mean free path {mean} cm");
    }

    #[test]
    fn directions_are_isotropic() {
        let rng = RandGenerator::new();
        rng.srand(2);
        let n = 200_000;
        let (mut sum, mut z2) = (DVec3::ZERO, 0.0);
        for _ in 0..n {
            let d = isotropic_direction(&rng);
            assert!((d.length() - 1.0).abs() < 1e-12);
            sum += d;
            z2 += d.z * d.z;
        }
        let mean = sum / n as f64;
        assert!(mean.abs().max_element() < 0.006, "mean direction {mean}");
        assert!((z2 / n as f64 - 1.0 / 3.0).abs() < 0.003, "mean z^2 {}", z2 / n as f64);
    }

    #[test]
    fn fission_releases_two_or_three_neutrons_on_average_nu() {
        let rng = RandGenerator::new();
        rng.srand(3);
        let n = 200_000;
        let mut total = 0;
        for _ in 0..n {
            let k = neutrons_per_fission(&rng, NU);
            assert!(k == 2 || k == 3, "{k} neutrons");
            total += k;
        }
        assert!((total as f64 / n as f64 - NU).abs() < 0.005);
    }

    #[test]
    fn pure_absorber_loses_neutrons_through_the_surface_as_predicted() {
        // Escape probability of a uniform absorbing sphere with x = Sigma_t R:
        //   P = 3 / (8 x^3) (2 x^2 - 1 + (1 + 2x) e^(-2x)).
        let absorber = || Material { sigma_f: 0.0, sigma_c: 1.0, sigma_s: 0.0, nu: NU };
        for (radius, expected) in [(1.0, 0.527252), (2.0, 0.332418), (3.0, 0.236352)] {
            let mut sim = ChainReaction::with(absorber(), radius, 200_000, 4);
            sim.settle = 0.0;
            sim.step(1e6);
            assert!(sim.is_over());
            let escaped = sim.tally.escapes as f64 / sim.tally.histories() as f64;
            assert!((escaped - expected).abs() < 0.005, "radius {radius}: {escaped}, expected {expected}");
        }
    }

    #[test]
    fn population_control_keeps_the_true_population() {
        for start in [5000, 700] {
            let mut sim = ChainReaction::with(Material::u235(), 6.0, start, 5);
            sim.target = 2000;
            let before = sim.relative_population();
            sim.population_control();
            assert_eq!(sim.neutrons.len(), 2000);
            assert!((sim.relative_population() / before - 1.0).abs() < 1e-9);
            assert!(sim.neutrons.iter().all(|neutron| neutron.pos.length() <= 6.0));
        }
    }

    #[test]
    fn same_seed_gives_same_run() {
        let go = || {
            let mut sim = ChainReaction::with(Material::u235(), 8.0, N_TARGET, 9);
            sim.settle = 0.0;
            for _ in 0..200 {
                sim.step(0.5);
            }
            sim
        };
        let (a, b) = (go(), go());
        assert_eq!(a.neutrons.len(), b.neutrons.len());
        assert_eq!(a.tally, b.tally);
        assert_eq!(a.neutrons[0].pos, b.neutrons[0].pos);
    }

    #[test]
    fn infinite_medium_has_k_infinity_and_alpha_infinity() {
        let sim = run(Material::u235(), 4000.0, 10, 20, 600);
        let m = &sim.material;
        let (k, alpha) = (sim.tally.k(m.nu), sim.tally.alpha(m.nu));
        assert!((k - 2.4604).abs() < 0.015, "k {k}");
        assert!((alpha - 0.12661).abs() < 0.002, "alpha {alpha}");
    }

    #[test]
    fn sood_benchmark_is_critical_at_its_published_radius() {
        let sim = run(sood(), 7.428998, 11, 50, 1200);
        let (k, alpha) = (sim.tally.k(2.70), sim.tally.alpha(2.70));
        assert!((k - 1.0).abs() < 0.010, "k {k}");
        assert!(alpha.abs() < 0.0026, "alpha {alpha}");
    }

    #[test]
    fn default_data_is_critical_at_the_one_group_radius() {
        let k = k_at(THEORY_RADIUS, 12, 1200);
        assert!((k - 1.0).abs() < 0.010, "k {k}");
    }

    #[test]
    fn three_percent_either_side_of_critical_is_sub_and_supercritical() {
        let (below, above) = (k_at(THEORY_RADIUS * 0.97, 13, 600), k_at(THEORY_RADIUS * 1.03, 14, 600));
        assert!(below < 0.99, "k below {below}");
        assert!(above > 1.01, "k above {above}");
    }

    #[test]
    fn k_rises_with_radius_and_crosses_one_near_the_critical_size() {
        let expected = [(5.0, 0.634), (6.0, 0.786), (7.0, 0.932), (8.0, 1.070), (9.0, 1.194), (10.0, 1.309), (11.0, 1.412)];
        let mut last = 0.0;
        for (radius, k_expected) in expected {
            let k = k_at(radius, 15, 300);
            assert!((k - k_expected).abs() < 0.02, "radius {radius}: k {k}, expected {k_expected}");
            assert!(k > last, "k not increasing at {radius}");
            last = k;
        }
        // 1 is crossed between 7 and 8 cm: expected values straddle it there.
        assert!(k_at(7.0, 16, 300) < 1.0 && k_at(8.0, 16, 300) > 1.0);
    }

    #[test]
    fn growth_rate_matches_the_tally_and_the_exact_value() {
        for (radius, expected) in [(9.0, 0.03472), (6.0, -0.05797)] {
            let mut sim = ChainReaction::with(Material::u235(), radius, N_TARGET, 17);
            sim.settle = 0.0;
            for _ in 0..50 {
                sim.step(1.0);
            }
            sim.tally = Tally::default();
            let start = sim.relative_population().ln();
            for _ in 0..1200 {
                sim.step(1.0);
            }
            let from_population = (sim.relative_population().ln() - start) / 1200.0;
            let alpha = sim.tally.alpha(sim.material.nu);
            assert!((alpha - expected).abs() < 0.002, "radius {radius}: alpha {alpha}, expected {expected}");
            assert!((from_population - alpha).abs() < 0.001, "radius {radius}: population {from_population}, tally {alpha}");
        }
    }
}
