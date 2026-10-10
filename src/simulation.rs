use macroquad::prelude::Rect;

/// A physics simulation that main.rs can run, pause and draw.
///
/// Time is in the simulation's own units (seconds, years, ...). Only `step`, `is_over`,
/// `world_bounds` and `draw` are required; the rest have sensible defaults.
pub trait Simulation {
    /// Advances the physics by `dt`. Only called while unpaused.
    fn step(&mut self, dt: f64);

    /// True once the simulation has finished. The final state stays on screen until the user restarts.
    fn is_over(&self) -> bool;

    /// The part of the world to show, in world units with y pointing up.
    /// It is scaled to fit the window, keeping the aspect ratio, so it can be resized freely.
    fn world_bounds(&self) -> Rect;

    /// Draws the current state in world coordinates. Called every frame, even while paused.
    /// Text goes in `hud` instead, since it can't be scaled with the world.
    fn draw(&self);

    /// Physics time step. Smaller means more accurate but slower.
    fn dt(&self) -> f64 {
        1.0 / 120.0
    }

    /// Simulated time per second of real time (1.0 is real time).
    fn time_scale(&self) -> f64 {
        1.0
    }

    /// Called every frame, even while paused, for simulation-specific keys.
    fn handle_input(&mut self) {}

    /// Lines of text (readouts, controls) drawn at the top-left of the screen.
    fn hud(&self) -> Vec<String> {
        Vec::new()
    }
}
