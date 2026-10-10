use macroquad::prelude::*;

use crate::simulation::Simulation;

/// Simulation units: distances in AU-like units, time in years-ish, with G*M_star = 1.
/// This keeps the numbers small and the inner planets fast enough to watch.
const GM: f32 = 1.0; // G * star mass
const SCALE: f32 = 55.0; // pixels per simulation unit
const CENTER_X: f32 = 450.0; // star position on screen
const CENTER_Y: f32 = 300.0;
const STAR_RADIUS_PX: f32 = 12.0;
const PLANET_RADIUS_PX: f32 = 5.0;
const TRAIL_LEN: usize = 400;

struct Planet {
    pos: Vec2,
    v: Vec2,
    color: Color,
    trail: Vec<Vec2>,
}

/// Eight planets on circular initial orbits around a fixed star.
pub struct Planets {
    planets: Vec<Planet>,
    t: f32,
}

impl Planets {
    pub fn new() -> Self {
        // Radii chosen to fill the window; angles spread so planets don't all line up.
        let radii = [0.9, 1.4, 1.9, 2.5, 3.1, 3.8, 4.5, 5.2];
        let colors = [SKYBLUE, ORANGE, LIME, GOLD, PINK, VIOLET, RED, BEIGE];
        let planets = radii
            .iter()
            .zip(colors.iter())
            .enumerate()
            .map(|(i, (&r, &color))| {
                let theta = i as f32 * std::f32::consts::TAU / radii.len() as f32;
                let pos = vec2(r * theta.cos(), r * theta.sin());
                // Circular-orbit speed, perpendicular to the radius (counter-clockwise).
                let speed = (GM / r).sqrt();
                let v = vec2(-theta.sin(), theta.cos()) * speed;
                Planet { pos, v, color, trail: Vec::with_capacity(TRAIL_LEN) }
            })
            .collect();
        Self { planets, t: 0.0 }
    }
}

impl Simulation for Planets {
    fn step(&mut self, dt: f32) {
        for planet in &mut self.planets {
            // Gravitational acceleration toward the star at the origin: a = -GM * r_hat / |r|^2.
            let r = planet.pos;
            let r2 = r.length_squared();
            let a = -r * (GM / (r2 * r2.sqrt()));
            // Semi-implicit Euler: update v first, then position with the new v.
            planet.v += a * dt;
            planet.pos += planet.v * dt;

            if planet.trail.len() == TRAIL_LEN {
                planet.trail.remove(0);
            }
            planet.trail.push(planet.pos);
        }
        self.t += dt;
    }

    fn is_over(&self) -> bool {
        false // runs indefinitely
    }

    fn draw(&self) {
        let center = vec2(CENTER_X, CENTER_Y);

        // Star.
        draw_circle(center.x, center.y, STAR_RADIUS_PX, YELLOW);

        for planet in &self.planets {
            // Trail: fade from dark to the planet's color.
            let n = planet.trail.len();
            for (i, window) in planet.trail.windows(2).enumerate() {
                let alpha = (i as f32 + 1.0) / n as f32;
                let c = Color::new(planet.color.r, planet.color.g, planet.color.b, alpha * 0.8);
                let p0 = center + window[0] * SCALE;
                let p1 = center + window[1] * SCALE;
                draw_line(p0.x, p0.y, p1.x, p1.y, 1.5, c);
            }
            let screen = center + planet.pos * SCALE;
            draw_circle(screen.x, screen.y, PLANET_RADIUS_PX, planet.color);
        }

        draw_text(&format!("t = {:.1}", self.t), 20.0, 60.0, 28.0, WHITE);
    }
}
