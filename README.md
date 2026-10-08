## Phyfun

Physics simulations, written for fun in Rust with [macroquad](https://github.com/not-fl3/macroquad).

### Running

```
cargo run -- <simulation name or number>
```

Run without an argument to list the simulations. A simulation starts paused; press `SPACE` to pause and unpause.

### Adding a simulation

1. Create `src/simulations/<name>.rs` with a struct implementing `Simulation` (see `src/simulation.rs`).
2. Register it at the end of `SIMULATIONS` in `src/simulations/mod.rs`. Its number is its position in that list.
