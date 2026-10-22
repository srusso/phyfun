## Phyfun

Physics simulations, written for fun in Rust with [macroquad](https://github.com/not-fl3/macroquad).

### Simulations

| # | Name | What it shows |
|---|------|---------------|
| 1 | `slope` | A block sliding down a frictionless slope. |
| 2 | `planets` | The Sun, the eight planets, Pluto and the Moon as a Newtonian N-body system, with Earth's precessing axis and the Moon's tides. |
| 3 | `double-pendulum` | Eight double pendulums released a millionth of a radian apart, fanning out through deterministic chaos. |
| 4 | `blackhole` | Particles and light on exact geodesics around a Schwarzschild black hole: orbit precession, the ISCO and the shadow. |
| 5 | `ising` | The 2D Ising ferromagnet with Metropolis updates, compared against Onsager's exact solution. |
| 6 | `chain-reaction` | A Monte Carlo neutron chain reaction in a bare sphere of uranium-235. |
| 7 | `top` | Rigid bodies turning about a fixed point: a tumbling paperback and spinning tops, in 3D. |
| 8 | `galaxies` | Two disk galaxies swinging past each other and pulling out tidal bridges and tails. |
| 9 | `argon` | Molecular dynamics of a 2D argon monolayer with a heat bath, melting from a crystal into a liquid. |
| 10 | `wavepacket` | An electron's wave function, solved with the Schrödinger equation, going through a double slit and other scenes. |
| 11 | `light` | A laser beam meeting glass, solved from Maxwell's equations on a grid (FDTD). |
| 12 | `vortex-street` | Water flowing past a rod at Reynolds number 100, shedding a von Kármán vortex street (lattice Boltzmann). |

### Running

```
cargo run -- <simulation name or number>
```

Run without an argument to list the simulations. A simulation starts paused; press `SPACE` to pause and unpause.

### Adding a simulation

1. Create `src/simulations/<name>.rs` with a struct implementing `Simulation` (see `src/simulation.rs`). Draw in world coordinates (`world_bounds`) and put text in `hud`; `dt`, `time_scale` and `handle_input` are optional.
2. Register it at the end of `SIMULATIONS` in `src/simulations/mod.rs`. Its number is its position in that list.
3. Add it to the table under **Simulations** above.
