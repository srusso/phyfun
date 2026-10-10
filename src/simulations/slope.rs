use macroquad::prelude::*;

use crate::simulation::Simulation;

const G: f32 = 9.81; // m/s^2
const ANGLE: f32 = 30.0 * std::f32::consts::PI / 180.0; // slope angle, rad
const SLOPE_LENGTH: f32 = 10.0; // m, along the surface
const BLOCK_SIZE: f32 = 0.6; // m, side of the square block

/// A block sliding down a frictionless slope, starting at rest.
pub struct Slope {
    /// Distance of the block's center from the top of the slope, along the surface.
    s: f32,
    /// Speed along the slope.
    v: f32,
    t: f32,
}

/// Where the block's center is when it reaches the bottom.
const END: f32 = SLOPE_LENGTH - BLOCK_SIZE / 2.0;

impl Slope {
    pub fn new() -> Self {
        Self { s: BLOCK_SIZE / 2.0, v: 0.0, t: 0.0 }
    }
}

impl Simulation for Slope {
    fn step(&mut self, dt: f64) {
        let dt = dt as f32;
        // Gravity's component along the slope; the normal force cancels the rest.
        let a = G * ANGLE.sin();
        // Semi-implicit Euler: update v first, then s with the new v.
        self.v += a * dt;
        self.s += self.v * dt;
        self.t += dt;
        self.s = self.s.min(END); // don't poke through the bottom
    }

    fn is_over(&self) -> bool {
        self.s >= END
    }

    fn world_bounds(&self) -> Rect {
        Rect::new(-1.0, -1.0, 11.0, 7.5)
    }

    fn draw(&self) {
        // World y points up, so the slope descends to the right.
        let down = vec2(ANGLE.cos(), -ANGLE.sin()); // unit vector along the surface
        let up = vec2(ANGLE.sin(), ANGLE.cos()); // unit normal, pointing away from the slope

        let top = vec2(0.0, SLOPE_LENGTH * ANGLE.sin());
        let bottom = top + down * SLOPE_LENGTH;
        draw_triangle(top, bottom, vec2(top.x, bottom.y), DARKGRAY);

        // Block corners: bottom edge on the surface, centered at distance `s`.
        let on_surface = top + down * self.s;
        let half = down * (BLOCK_SIZE / 2.0);
        let height = up * BLOCK_SIZE;
        let (a, b) = (on_surface - half, on_surface + half);
        let (c, d) = (b + height, a + height);
        draw_triangle(a, b, c, ORANGE);
        draw_triangle(a, c, d, ORANGE);
    }

    fn hud(&self) -> Vec<String> {
        vec![format!("t = {:.2} s", self.t), format!("v = {:.2} m/s", self.v)]
    }
}
