mod simulation;
mod simulations;

use macroquad::prelude::*;
use simulation::Simulation;

/// Fixed physics time step, independent of the frame rate.
const DT: f32 = 1.0 / 120.0;

fn main() {
    let Some(query) = std::env::args().nth(1) else {
        usage("missing simulation");
    };
    let Some((number, entry)) = simulations::find(&query) else {
        usage(&format!("unknown simulation '{query}'"));
    };

    let title = format!("phyfun - {number} {}", entry.name);
    let conf = Conf {
        window_title: title.clone(),
        window_width: 900,
        window_height: 600,
        ..Default::default()
    };
    macroquad::Window::from_config(conf, run((entry.create)(), title));
}

fn usage(error: &str) -> ! {
    eprintln!("error: {error}\n");
    eprintln!("usage: cargo run -- <simulation name or number>\n");
    eprintln!("simulations:\n{}", simulations::list());
    std::process::exit(1);
}

async fn run(mut sim: Box<dyn Simulation>, title: String) {
    let mut paused = true; // start paused
    let mut lag = 0.0; // real time not yet simulated

    loop {
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        if !paused {
            // Clamp so a long stall (e.g. dragging the window) doesn't cause a burst of steps.
            lag += get_frame_time().min(0.1);
            while lag >= DT {
                sim.step(DT);
                lag -= DT;
            }
        }

        clear_background(BLACK);
        sim.draw();
        draw_text(&title, 20.0, 30.0, 28.0, GRAY);
        if paused {
            draw_text("PAUSED - press SPACE", 20.0, 580.0, 28.0, YELLOW);
        }
        next_frame().await;
    }
}
