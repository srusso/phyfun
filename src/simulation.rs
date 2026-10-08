/// A physics simulation that main.rs can run, pause and draw.
pub trait Simulation {
    /// Advances the physics by `dt` seconds. Only called while unpaused.
    fn step(&mut self, dt: f32);

    /// Draws the current state. Called every frame, even while paused.
    fn draw(&self);
}
