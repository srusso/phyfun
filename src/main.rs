mod simulation;
mod simulations;

use macroquad::prelude::*;
use simulation::Simulation;

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
    macroquad::Window::from_config(conf, run(entry.create, title));
}

fn usage(error: &str) -> ! {
    eprintln!("error: {error}\n");
    eprintln!("usage: cargo run -- <simulation name or number>\n");
    eprintln!("simulations:\n{}", simulations::list());
    std::process::exit(1);
}

/// A camera showing `bounds` (y up), grown on one axis to match the window's aspect ratio.
fn camera_for(bounds: Rect) -> Camera2D {
    let aspect = screen_width() / screen_height();
    let (mut w, mut h) = (bounds.w, bounds.h);
    if w / h < aspect {
        w = h * aspect;
    } else {
        h = w / aspect;
    }
    Camera2D {
        target: vec2(bounds.x + bounds.w / 2.0, bounds.y + bounds.h / 2.0),
        zoom: vec2(2.0 / w, 2.0 / h),
        ..Default::default()
    }
}

async fn run(create: fn() -> Box<dyn Simulation>, title: String) {
    let mut sim = create();
    let mut paused = true; // start paused
    let mut lag = 0.0; // real time not yet simulated

    // FPS shown on screen: frames counted over a window and refreshed when it ends.
    // The first window is short so a value shows up almost immediately.
    let mut fps = 0;
    let mut fps_frames = 0;
    let mut fps_elapsed = 0.0;
    let mut fps_window = 0.1; // s

    loop {
        if is_key_pressed(KeyCode::Q) {
            return;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        sim.handle_input();

        if !paused {
            // Clamp so a long stall (e.g. dragging the window) doesn't cause a burst of steps.
            lag += get_frame_time().min(0.1) as f64 * sim.time_scale();
            let dt = sim.dt();
            while lag >= dt {
                sim.step(dt);
                lag -= dt;
            }
        }

        if sim.is_over() {
            // Back to the initial state, paused.
            sim = create();
            paused = true;
            lag = 0.0;
        }

        clear_background(BLACK);
        set_camera(&camera_for(sim.world_bounds()));
        sim.draw();
        set_default_camera();

        draw_text(&title, 20.0, 30.0, 28.0, GRAY);
        for (i, line) in sim.hud().iter().enumerate() {
            draw_text(line, 20.0, 60.0 + 30.0 * i as f32, 28.0, WHITE);
        }
        if paused {
            draw_text("PAUSED - press SPACE", 20.0, 580.0, 28.0, YELLOW);
        }
        fps_frames += 1;
        fps_elapsed += get_frame_time();
        if fps_elapsed >= fps_window {
            fps = (fps_frames as f32 / fps_elapsed).round() as u32;
            fps_frames = 0;
            fps_elapsed = 0.0;
            fps_window = 1.0; // then once per second
        }
        let fps = format!("{fps} FPS");
        let size = measure_text(&fps, None, 24, 1.0);
        draw_text(&fps, screen_width() - size.width - 20.0, screen_height() - 20.0, 24.0, GRAY);
        next_frame().await;
    }
}
