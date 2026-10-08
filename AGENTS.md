# Agent instructions

Physics simulations and experiments, written for fun.

## Stack
- Language: Rust
- Graphics: [macroquad](https://github.com/not-fl3/macroquad)
- Vector calculus / linear algebra: [nalgebra](https://nalgebra.org), only if actually needed. Don't add it for simple 1D/2D math that plain `f32`/`f64` or macroquad's own vector types handle.

## Code style
- Keep code easy to read and use.
- Add short, descriptive comments for non-obvious code (the physics, the integrator, unit conventions). Don't comment what the code already says.
- Short names are fine for standard physics symbols: `v` for velocity, `a` for acceleration, `t` for time, `dt` for the time step, `m` for mass, `g` for gravity.
- Otherwise use descriptive names for the things being simulated. In a simulation with a slope and a block sliding down it, call them `block` and `slope`, not `b` and `s`.
