use macroquad::prelude::*;

use crate::simulation::Simulation;

const G: f32 = 9.81; // m/s^2
const ANGLE: f32 = 30.0 * std::f32::consts::PI / 180.0; // slope angle, rad
const SLOPE_LENGTH: f32 = 10.0; // m, along the surface
const BLOCK_SIZE: f32 = 0.6; // m, side of the square block
const SCALE: f32 = 80.0; // pixels per meter

/// A block sliding down a frictionless slope, starting at rest.
pub struct Slope {
    /// Distance of the block's center from the top of the slope, along the surface.
    s: f32,
    /// Speed along the slope.
    v: f32,
    t: f32,
}

impl Slope {
    pub fn new() -> Self {
        Self { s: BLOCK_SIZE / 2.0, v: 0.0, t: 0.0 }
    }
}

impl Simulation for Slope {
    fn step(&mut self, dt: f32) {
        let end = SLOPE_LENGTH - BLOCK_SIZE / 2.0;
        if self.s >= end {
            return; // reached the bottom, stay put
        }
        // Gravity's component along the slope; the normal force cancels the rest.
        let a = G * ANGLE.sin();
        // Semi-implicit Euler: update v first, then s with the new v.
        self.v += a * dt;
        self.s += self.v * dt;
        self.t += dt;
        if self.s >= end {
            self.s = end;
            self.v = 0.0;
        }
    }

    fn draw(&self) {
        // Screen y points down, so the slope descends to the right.
        let down = vec2(ANGLE.cos(), ANGLE.sin()); // unit vector along the surface
        let up = vec2(ANGLE.sin(), -ANGLE.cos()); // unit normal, pointing away from the slope

        let top = vec2(100.0, 120.0);
        let bottom = top + down * SLOPE_LENGTH * SCALE;
        draw_triangle(top, bottom, vec2(top.x, bottom.y), DARKGRAY);

        // Block corners: bottom edge on the surface, centered at distance `s`.
        let on_surface = top + down * self.s * SCALE;
        let half = down * (BLOCK_SIZE / 2.0 * SCALE);
        let height = up * (BLOCK_SIZE * SCALE);
        let (a, b) = (on_surface - half, on_surface + half);
        let (c, d) = (b + height, a + height);
        draw_triangle(a, b, c, ORANGE);
        draw_triangle(a, c, d, ORANGE);

        draw_text(format!("t = {:.2} s", self.t), 20.0, 60.0, 28.0, WHITE);
        draw_text(format!("v = {:.2} m/s", self.v), 20.0, 90.0, 28.0, WHITE);
    }
}
