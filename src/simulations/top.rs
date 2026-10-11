use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use macroquad::prelude::*;

use crate::simulation::Simulation;

// Units: SI (m, kg, s, rad). World frame: z up, gravity along -z. Body frame: the principal axes
// e1, e2, e3 of the body, with the origin at the fixed point (the centre of mass for the book,
// the pivot for the top).

const G: f64 = 9.80665; // m/s^2, standard gravity
const ALUMINIUM_DENSITY: f64 = 2700.0; // kg/m^3

/// Time step: Yoshida's fourth-order composition of three symmetric steps.
const DT: f64 = 0.001;
/// Yoshida (1990): `W1 = 1 / (2 - 2^(1/3))`, and `W0 = 1 - 2 W1` (negative: a step backwards).
const YOSHIDA_W1: f64 = 1.351_207_191_959_657_8;
const YOSHIDA_W0: f64 = -1.702_414_383_919_315_5;

// The book: 200 sheets of A5 paper (80 g/m^2, 0.1 mm thick) as a uniform cuboid, covers ignored.
const BOOK_SIZE: DVec3 = DVec3::new(0.210, 0.148, 0.020); // m, along e1, e2, e3
const SHEETS: f64 = 200.0;
const PAPER_KG_PER_M2: f64 = 0.080;

/// The top: three coaxial solid aluminium cylinders along e3, as (z from, z to, radius) in m, with
/// the pivot at z = 0: lower stem, disc, upper stem. The same list is drawn.
const TOP_PARTS: [(f64, f64, f64); 3] = [(0.0, 0.025, 0.003), (0.025, 0.035, 0.050), (0.035, 0.080, 0.003)];

const PRESET_SPEEDS: [f64; 7] = [1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25];
const MIN_SPEED: f64 = 1.0 / 16.0;
const MAX_SPEED: f64 = 2.0;

// Drawing.
const TRAIL_ANGLE: f64 = 0.02; // rad of pointer motion between trail points
const TRAIL_LEN: usize = 1500;
const VIEW_TURN_RATE: f32 = 1.5; // rad/s
const START_VIEW_ANGLE: f32 = -60.0 * std::f32::consts::PI / 180.0;
const ELEVATION: f32 = 20.0 * std::f32::consts::PI / 180.0;
const FOV: f32 = 30.0 * std::f32::consts::PI / 180.0;

/// Mass and principal moments of inertia of the book.
fn book_inertia() -> (f64, DVec3) {
    let (a, b, c) = (BOOK_SIZE.x, BOOK_SIZE.y, BOOK_SIZE.z);
    let m = SHEETS * a * b * PAPER_KG_PER_M2;
    // Uniform cuboid: I1 = m (b^2 + c^2) / 12 and cyclic.
    (m, dvec3(b * b + c * c, a * a + c * c, a * a + b * b) * (m / 12.0))
}

/// Mass, distance from the pivot to the centre of mass, and moments of inertia about the pivot.
/// (`m` and `l` are only checked by the tests.)
#[cfg_attr(not(test), allow(dead_code))]
struct TopProperties {
    m: f64,
    l: f64,
    /// m g l, the torque of gravity on a top lying on its side.
    weight_moment: f64,
    /// About the symmetry axis.
    i3: f64,
    /// About any axis through the pivot perpendicular to it.
    i1: f64,
}

fn top_properties() -> TopProperties {
    let (mut m, mut first_moment, mut i3, mut i1) = (0.0, 0.0, 0.0, 0.0);
    for (z_from, z_to, r) in TOP_PARTS {
        let (h, zc) = (z_to - z_from, (z_from + z_to) / 2.0);
        let mc = ALUMINIUM_DENSITY * PI * r * r * h;
        m += mc;
        first_moment += mc * zc;
        i3 += 0.5 * mc * r * r;
        // About a transverse axis through the pivot (parallel-axis theorem).
        i1 += mc * (3.0 * r * r + h * h) / 12.0 + mc * zc * zc;
    }
    let l = first_moment / m;
    TopProperties { m, l, weight_moment: m * G * l, i3, i1 }
}

/// Spin rate about the symmetry axis above which a top started upright stays upright, rad/s.
fn sleeping_threshold() -> f64 {
    let top = top_properties();
    2.0 * (top.weight_moment * top.i1).sqrt() / top.i3
}

/// Slow steady precession rate (rad/s) at tilt `tilt` from the vertical, for spin `spin` about the
/// symmetry axis: the slow root of `I1 W^2 cos(tilt) - I3 w3 W + m g l = 0`.
fn steady_precession(tilt: f64, spin: f64) -> f64 {
    let top = top_properties();
    let (cos, i3_spin) = (tilt.cos(), top.i3 * spin);
    if cos.abs() < 1e-12 {
        return top.weight_moment / i3_spin; // lying on its side the equation is linear
    }
    (i3_spin - (i3_spin * i3_spin - 4.0 * top.i1 * top.weight_moment * cos).sqrt()) / (2.0 * top.i1 * cos)
}

/// Largest tilt (rad) of a top released at rest in precession (W = 0) at tilt `tilt` with spin
/// `spin`: it nods between `tilt` and this.
fn released_tilt_limit(tilt: f64, spin: f64) -> f64 {
    let top = top_properties();
    let alpha = 2.0 * top.weight_moment / top.i1;
    let p = top.i3 * spin / top.i1;
    let u0 = tilt.cos();
    let u1 = (p * p - (p.powi(4) - 4.0 * alpha * (p * p * u0 - alpha)).sqrt()) / (2.0 * alpha);
    u1.acos()
}

/// Arithmetic-geometric mean.
fn agm(mut x: f64, mut y: f64) -> f64 {
    while (x - y).abs() >= 1e-15 * x {
        (x, y) = ((x + y) / 2.0, (x * y).sqrt());
    }
    x
}

/// Time between sign changes of L2 for a torque-free asymmetric body (Landau & Lifshitz,
/// Mechanics, section 37), from the Jacobi elliptic solution.
fn flip_interval(inertia: DVec3, momentum: DVec3) -> f64 {
    let (i1, i2, i3) = (inertia.x, inertia.y, inertia.z);
    let ll = momentum.length_squared();
    let twice_energy = momentum.x * momentum.x / i1 + momentum.y * momentum.y / i2 + momentum.z * momentum.z / i3;
    let (n, k2) = if ll > twice_energy * i2 {
        (
            ((i3 - i2) * (ll - twice_energy * i1) / (i1 * i2 * i3)).sqrt(),
            (i2 - i1) * (twice_energy * i3 - ll) / ((i3 - i2) * (ll - twice_energy * i1)),
        )
    } else {
        // The same with I1 and I3 swapped.
        (
            ((i1 - i2) * (ll - twice_energy * i3) / (i1 * i2 * i3)).sqrt(),
            (i2 - i3) * (twice_energy * i1 - ll) / ((i1 - i2) * (ll - twice_energy * i3)),
        )
    };
    let k = PI / (2.0 * agm(1.0, (1.0 - k2).sqrt()));
    2.0 * k / n
}

/// A triangle of the body's surface, in the body frame.
struct Triangle {
    corners: [Vec3; 3],
    normal: Vec3,
    color: Color,
}

/// A cuboid with the given half sizes; `colors` are for the faces +e1, -e1, +e2, -e2, +e3, -e3.
fn cuboid(half: Vec3, colors: [Color; 6]) -> Vec<Triangle> {
    let mut triangles = Vec::new();
    for (face, color) in colors.into_iter().enumerate() {
        let (axis, sign) = (face / 2, if face % 2 == 0 { 1.0 } else { -1.0 });
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let corner = |a: f32, b: f32| {
            let mut p = Vec3::ZERO;
            p[axis] = sign * half[axis];
            p[u] = a * half[u];
            p[v] = b * half[v];
            p
        };
        let mut normal = Vec3::ZERO;
        normal[axis] = sign;
        let (c00, c10, c11, c01) = (corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0));
        triangles.push(Triangle { corners: [c00, c10, c11], normal, color });
        triangles.push(Triangle { corners: [c00, c11, c01], normal, color });
    }
    triangles
}

/// A cylinder along e3 from `z_from` to `z_to`; `side_color` gives each side's colour.
fn cylinder(z_from: f64, z_to: f64, radius: f64, sides: usize, side_color: impl Fn(usize) -> Color, cap_color: Color) -> Vec<Triangle> {
    let (z0, z1, r) = (z_from as f32, z_to as f32, radius as f32);
    let rim = |i: usize| {
        let angle = TAU as f32 * i as f32 / sides as f32;
        (angle.cos(), angle.sin())
    };
    let mut triangles = Vec::new();
    for i in 0..sides {
        let ((c0, s0), (c1, s1)) = (rim(i), rim(i + 1));
        let (a, b) = (vec3(r * c0, r * s0, z0), vec3(r * c1, r * s1, z0));
        let (c, d) = (vec3(r * c1, r * s1, z1), vec3(r * c0, r * s0, z1));
        let normal = vec3(c0 + c1, s0 + s1, 0.0).normalize();
        let color = side_color(i);
        triangles.push(Triangle { corners: [a, b, c], normal, color });
        triangles.push(Triangle { corners: [a, c, d], normal, color });
        triangles.push(Triangle { corners: [vec3(0.0, 0.0, z1), d, c], normal: Vec3::Z, color: cap_color });
        triangles.push(Triangle { corners: [vec3(0.0, 0.0, z0), b, a], normal: -Vec3::Z, color: cap_color });
    }
    triangles
}

/// A rigid body turning about a fixed point: a paperback in free fall (centre of mass fixed) or a
/// top on a pivot under gravity. Both are Euler's equations for the angular momentum `momentum`
/// (body frame) plus a quaternion `q` for the orientation; the top adds a torque.
/// Seven presets on keys 1-7, drawn in 3D (LEFT/RIGHT turn the view).
/// Not modelled: friction at the pivot or in the air (a real top slows and a sleeping top wakes
/// up), a tip that slides or rolls on a table, a book that flexes or has covers (and its fall),
/// and the Earth's rotation.
pub struct Top {
    name: &'static str,
    /// Principal moments of inertia, kg m^2.
    inertia: DVec3,
    /// m g l, N m: the torque scale of gravity (0 for the book).
    weight_moment: f64,
    /// Rotates body vectors into the world frame.
    q: DQuat,
    /// Angular momentum in the body frame.
    momentum: DVec3,
    t: f64,
    speed: f64,
    /// The body axis (0, 1 or 2) that the preset spins about.
    spin_axis: usize,
    energy0: f64,
    world_momentum0: DVec3,
    /// Starting values of L3, of L along the vertical (conserved for the top) and of |w|.
    l3_0: f64,
    lz0: f64,
    omega0: f64,
    /// Sign changes of `momentum[spin_axis]`, the time of the last one and the time between the
    /// last two.
    flips: u32,
    last_flip_t: Option<f64>,
    flip_interval: Option<f64>,
    max_tilt: f64,
    /// Unwrapped azimuth (rad) of the symmetry axis, and the raw angle it was last updated from.
    azimuth: f64,
    last_azimuth: f64,
    azimuth0: f64,
    /// Prediction shown next to the measurement: flip interval (s) or largest tilt (rad).
    theory: Option<f64>,
    // Drawing.
    shape: Vec<Triangle>,
    bounds: Rect,
    view_angle: f32,
    /// Unit vectors (world frame) along the pointer axis, for the trail.
    trail: VecDeque<Vec3>,
}

impl Top {
    pub fn new() -> Self {
        Self::preset(1)
    }

    fn preset(number: usize) -> Self {
        let w0 = TAU;
        let mut top = match number {
            1 => Self::book("book: intermediate axis (flips)", 1, dvec3(0.01, 1.0, 0.01) * w0),
            2 => Self::book("book: long axis (stable)", 0, dvec3(1.0, 0.01, 0.01) * w0),
            3 => Self::book("book: thickness axis (stable)", 2, dvec3(0.01, 0.01, 1.0) * w0),
            4 => Self::top("top: released at 30 deg (cusps)", 30f64.to_radians(), 60.0, 0.0),
            5 => {
                let tilt = 30f64.to_radians();
                Self::top("top: steady precession", tilt, 60.0, steady_precession(tilt, 60.0))
            }
            6 => Self::top("top: sleeping", 1f64.to_radians(), 1.2 * sleeping_threshold(), 0.0),
            _ => Self::top("top: too slow to sleep", 1f64.to_radians(), 0.9 * sleeping_threshold(), 0.0),
        };
        top.speed = PRESET_SPEEDS[number.clamp(1, 7) - 1];
        top
    }

    /// The book spinning with angular velocity `w` (body frame) about the body axis `spin_axis`
    /// (plus a small disturbance on the others), with that axis starting vertical.
    fn book(name: &'static str, spin_axis: usize, w: DVec3) -> Self {
        let (_, inertia) = book_inertia();
        let mut axis = DVec3::ZERO;
        axis[spin_axis] = 1.0;
        let half = (BOOK_SIZE / 2.0).as_vec3();
        let shape = cuboid(half, [BEIGE, BEIGE, WHITE, DARKGRAY, RED, BLUE]);
        let mut top = Self::from_state(name, inertia, 0.0, DQuat::from_rotation_arc(axis, DVec3::Z), inertia * w, spin_axis, shape);
        top.bounds = Rect::new(-0.27, -0.18, 0.54, 0.36);
        top.theory = if spin_axis == 1 && w.x != 0.0 { Some(flip_interval(inertia, top.momentum)) } else { None };
        top
    }

    /// The top tilted `tilt` from the vertical (towards +x) with spin `spin` about its axis (the
    /// component of the angular velocity along e3) and precession rate `precession` about the vertical.
    fn top(name: &'static str, tilt: f64, spin: f64, precession: f64) -> Self {
        let props = top_properties();
        let momentum = dvec3(-props.i1 * precession * tilt.sin(), 0.0, props.i3 * spin);
        let mut shape = Vec::new();
        for (index, (z_from, z_to, radius)) in TOP_PARTS.into_iter().enumerate() {
            shape.extend(if index == 1 {
                // The disc has a darker quarter so the spin is visible.
                cylinder(z_from, z_to, radius, 24, |i| if i < 6 { MAROON } else { ORANGE }, ORANGE)
            } else {
                cylinder(z_from, z_to, radius, 24, |_| LIGHTGRAY, LIGHTGRAY)
            });
        }
        let mut top = Self::from_state(
            name,
            dvec3(props.i1, props.i1, props.i3),
            props.weight_moment,
            DQuat::from_rotation_y(tilt),
            momentum,
            2,
            shape,
        );
        top.bounds = Rect::new(-0.18, -0.06, 0.36, 0.24);
        top.theory = Some(if precession == 0.0 { released_tilt_limit(tilt, spin) } else { tilt });
        top
    }

    fn from_state(name: &'static str, inertia: DVec3, weight_moment: f64, q: DQuat, momentum: DVec3, spin_axis: usize, shape: Vec<Triangle>) -> Self {
        let mut top = Self {
            name,
            inertia,
            weight_moment,
            q,
            momentum,
            t: 0.0,
            speed: 1.0,
            spin_axis,
            energy0: 0.0,
            world_momentum0: DVec3::ZERO,
            l3_0: momentum.z,
            lz0: 0.0,
            omega0: 0.0,
            flips: 0,
            last_flip_t: None,
            flip_interval: None,
            max_tilt: 0.0,
            azimuth: 0.0,
            last_azimuth: 0.0,
            azimuth0: 0.0,
            theory: None,
            shape,
            bounds: Rect::new(-0.27, -0.18, 0.54, 0.36),
            view_angle: START_VIEW_ANGLE,
            trail: VecDeque::new(),
        };
        top.energy0 = top.energy();
        top.world_momentum0 = top.world_momentum();
        top.lz0 = top.momentum.dot(top.up());
        top.omega0 = top.angular_velocity().length();
        top.max_tilt = top.tilt();
        let axis = top.axis(2);
        top.last_azimuth = axis.y.atan2(axis.x);
        top.azimuth0 = top.last_azimuth;
        top
    }

    /// Angular velocity in the body frame.
    fn angular_velocity(&self) -> DVec3 {
        self.momentum / self.inertia
    }

    /// Kinetic plus potential energy.
    fn energy(&self) -> f64 {
        0.5 * self.momentum.dot(self.angular_velocity()) + self.weight_moment * self.up().z
    }

    /// Angular momentum in the world frame (conserved for the book).
    fn world_momentum(&self) -> DVec3 {
        self.q * self.momentum
    }

    /// Body axis `i` in the world frame.
    fn axis(&self, i: usize) -> DVec3 {
        let mut e = DVec3::ZERO;
        e[i] = 1.0;
        self.q * e
    }

    /// The world vertical seen from the body.
    fn up(&self) -> DVec3 {
        self.q.inverse() * DVec3::Z
    }

    /// Angle of the symmetry axis from the vertical. Uses `atan2`, which keeps its precision near
    /// upright where `acos` would not.
    fn tilt(&self) -> f64 {
        let axis = self.axis(2);
        axis.x.hypot(axis.y).atan2(axis.z)
    }

    /// Gravity's torque on the top for half a step: `L += h m g l (up.y, -up.x, 0)`.
    fn kick(&mut self, h: f64) {
        if self.weight_moment != 0.0 {
            let up = self.up();
            self.momentum += dvec3(up.y, -up.x, 0.0) * (h * self.weight_moment);
        }
    }

    /// The body turns by `angle` about its own axis `i`; seen from the body, `L` turns the other way.
    fn turn_about_axis(&mut self, i: usize, angle: f64) {
        let mut e = DVec3::ZERO;
        e[i] = 1.0;
        self.q *= DQuat::from_axis_angle(e, angle);
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let (sin, cos) = angle.sin_cos();
        (self.momentum[j], self.momentum[k]) = (
            self.momentum[j] * cos + self.momentum[k] * sin,
            -self.momentum[j] * sin + self.momentum[k] * cos,
        );
    }

    /// Torque-free rotation for time `h`. The kinetic energy is rewritten as
    ///   |L|^2 / (2 I2) + 1/2 (1/I1 - 1/I2) L1^2 + 1/2 (1/I3 - 1/I2) L3^2
    /// and each term moves the body by an exact rotation, so `q` and `L` turn together and the
    /// angular momentum in the world frame is conserved to rounding. For a symmetric top
    /// (I1 = I2) the first and last turns have zero angle and the result is exact.
    fn free_rotation(&mut self, h: f64) {
        let (i1, i2, i3) = (self.inertia.x, self.inertia.y, self.inertia.z);
        self.turn_about_axis(0, (1.0 / i1 - 1.0 / i2) * self.momentum.x * h / 2.0);
        self.turn_about_axis(2, (1.0 / i3 - 1.0 / i2) * self.momentum.z * h);
        self.turn_about_axis(0, (1.0 / i1 - 1.0 / i2) * self.momentum.x * h / 2.0);
        let size = self.momentum.length();
        if size > 0.0 {
            // Turn about L itself, which leaves L unchanged.
            self.q *= DQuat::from_axis_angle(self.momentum / size, size * h / i2);
        }
    }

    /// Second-order, time-symmetric step: half a kick, a free rotation, half a kick.
    fn base_step(&mut self, h: f64) {
        self.kick(h / 2.0);
        self.free_rotation(h);
        self.kick(h / 2.0);
    }
}

impl Simulation for Top {
    fn step(&mut self, dt: f64) {
        let before = self.momentum[self.spin_axis];
        // Fourth order from three second-order steps (Yoshida).
        self.base_step(YOSHIDA_W1 * dt);
        self.base_step(YOSHIDA_W0 * dt);
        self.base_step(YOSHIDA_W1 * dt);
        self.q = self.q.normalize(); // only removes rounding
        let t_before = self.t;
        self.t += dt;

        // Readouts. A flip is a sign change of L along the spin axis; the crossing time is
        // interpolated linearly inside the step.
        let after = self.momentum[self.spin_axis];
        if (before < 0.0) != (after < 0.0) {
            let crossing = t_before + dt * before / (before - after);
            if let Some(last) = self.last_flip_t {
                self.flip_interval = Some(crossing - last);
            }
            self.last_flip_t = Some(crossing);
            self.flips += 1;
        }
        self.max_tilt = self.max_tilt.max(self.tilt());
        let symmetry_axis = self.axis(2);
        let raw = symmetry_axis.y.atan2(symmetry_axis.x);
        self.azimuth += (raw - self.last_azimuth + PI).rem_euclid(TAU) - PI;
        self.last_azimuth = raw;

        let pointer = self.axis(self.spin_axis).as_vec3();
        if self.trail.back().is_none_or(|last| (last.dot(pointer) as f64) < TRAIL_ANGLE.cos()) {
            if self.trail.len() == TRAIL_LEN {
                self.trail.pop_front();
            }
            self.trail.push_back(pointer);
        }
    }

    fn is_over(&self) -> bool {
        false
    }

    fn dt(&self) -> f64 {
        DT
    }

    fn time_scale(&self) -> f64 {
        self.speed
    }

    fn world_bounds(&self) -> Rect {
        self.bounds
    }

    fn handle_input(&mut self) {
        let keys = [KeyCode::Key1, KeyCode::Key2, KeyCode::Key3, KeyCode::Key4, KeyCode::Key5, KeyCode::Key6, KeyCode::Key7];
        for (i, key) in keys.into_iter().enumerate() {
            if is_key_pressed(key) {
                let view_angle = self.view_angle;
                *self = Self::preset(i + 1);
                self.view_angle = view_angle;
            }
        }
        if is_key_pressed(KeyCode::Up) {
            self.speed = (self.speed * 2.0).min(MAX_SPEED);
        }
        if is_key_pressed(KeyCode::Down) {
            self.speed = (self.speed / 2.0).max(MIN_SPEED);
        }
        let turn = VIEW_TURN_RATE * get_frame_time();
        if is_key_down(KeyCode::Left) {
            self.view_angle -= turn;
        }
        if is_key_down(KeyCode::Right) {
            self.view_angle += turn;
        }
    }

    fn draw(&self) {
        // A 3D camera, set here so main.rs needs no change. Fitted to the bounds like the 2D one.
        let bounds = self.bounds;
        let aspect = screen_width() / screen_height();
        let view_height = bounds.h.max(bounds.w / aspect);
        let distance = view_height / (2.0 * (FOV / 2.0).tan());
        let target = vec3(0.0, 0.0, bounds.y + bounds.h / 2.0);
        let direction = vec3(ELEVATION.cos() * self.view_angle.cos(), ELEVATION.cos() * self.view_angle.sin(), ELEVATION.sin());
        set_camera(&Camera3D {
            position: target + direction * distance,
            target,
            up: Vec3::Z,
            fovy: FOV,
            z_near: distance / 10.0,
            z_far: distance * 10.0,
            ..Default::default()
        });

        // Body: one mesh, lit by a fixed light.
        let rotation = self.q.as_quat();
        let light = vec3(-0.3, -0.6, 0.75).normalize();
        let mut mesh = Mesh { vertices: Vec::new(), indices: Vec::new(), texture: None };
        for triangle in &self.shape {
            let shade = 0.35 + 0.65 * (rotation * triangle.normal).dot(light).max(0.0);
            let color = Color::new(triangle.color.r * shade, triangle.color.g * shade, triangle.color.b * shade, 1.0);
            for corner in triangle.corners {
                mesh.indices.push(mesh.vertices.len() as u16);
                mesh.vertices.push(Vertex::new2(rotation * corner, Vec2::ZERO, color));
            }
        }
        draw_mesh(&mesh);

        if self.weight_moment != 0.0 {
            // Floor grid at z = 0.
            for i in -3..=3 {
                let c = 0.05 * i as f32;
                draw_line_3d(vec3(c, -0.15, 0.0), vec3(c, 0.15, 0.0), DARKGRAY);
                draw_line_3d(vec3(-0.15, c, 0.0), vec3(0.15, c, 0.0), DARKGRAY);
            }
        }

        // Angular momentum (yellow) and angular velocity (sky blue), scaled to their starting size.
        let arrow = |vector: DVec3, start_size: f64, color: Color| {
            let tip = (vector / start_size.max(1e-30) * (0.4 * bounds.h as f64)).as_vec3();
            draw_line_3d(Vec3::ZERO, tip, color);
            draw_sphere(tip, 0.012 * bounds.h, None, color);
        };
        arrow(self.world_momentum(), self.world_momentum0.length(), YELLOW);
        arrow(self.q * self.angular_velocity(), self.omega0, SKYBLUE);

        // Pointer along the spin axis and the path of its tip.
        let length = 0.42 * bounds.h;
        let tip = self.axis(self.spin_axis).as_vec3() * length;
        draw_line_3d(Vec3::ZERO, tip, WHITE);
        let n = self.trail.len();
        for (i, (p0, p1)) in self.trail.iter().zip(self.trail.iter().skip(1)).enumerate() {
            let alpha = (i + 1) as f32 / n as f32;
            draw_line_3d(*p0 * length, *p1 * length, Color::new(1.0, 1.0, 1.0, alpha));
        }
    }

    fn hud(&self) -> Vec<String> {
        let mut lines = vec![
            format!("{}   (1-7 scene, LEFT/RIGHT view)", self.name),
            format!("t = {:.2} s   speed x{} (UP/DOWN)", self.t, self.speed),
            format!("energy drift = {:+.1e}", self.energy() / self.energy0 - 1.0),
        ];
        let tilt = self.tilt().to_degrees();
        if self.weight_moment == 0.0 {
            let drift = (self.world_momentum() - self.world_momentum0).length() / self.world_momentum0.length();
            let to_l = self.axis(self.spin_axis).angle_between(self.world_momentum()).to_degrees();
            lines.push(format!("angular momentum drift = {:.1e}", drift));
            lines.push(format!("spin axis {:.1} deg from L   flips: {}", to_l, self.flips));
            if let Some(theory) = self.theory {
                let measured = self.flip_interval.map_or("-".to_string(), |interval| format!("{interval:.3}"));
                lines.push(format!("every {} s (theory {:.3})", measured, theory));
            }
        } else {
            let lz = self.momentum.dot(self.up()) / self.lz0;
            let l3 = self.momentum.z / self.l3_0;
            lines.push(format!("Lz drift = {:+.1e}   L3 drift = {:+.1e}", lz - 1.0, l3 - 1.0));
            let theory = self.theory.map_or(0.0, f64::to_degrees);
            lines.push(format!("tilt {:.2} deg   max {:.2} (theory {:.2})", tilt, self.max_tilt.to_degrees(), theory));
            let props = top_properties();
            let precession = (self.azimuth - self.azimuth0) / self.t.max(1e-9);
            lines.push(format!(
                "precession {:.3} rad/s (fast-top est. {:.3})",
                precession,
                props.weight_moment / self.momentum.z
            ));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(top: &mut Top, seconds: f64) {
        for _ in 0..(seconds / DT).round() as usize {
            top.step(DT);
        }
    }

    fn close(value: f64, expected: f64, relative: f64, what: &str) {
        assert!((value - expected).abs() <= relative * expected.abs(), "{what}: {value}, expected {expected}");
    }

    /// Angle of the rotation taking `a` to `b`.
    fn orientation_error(a: DQuat, b: DQuat) -> f64 {
        let d = a.inverse() * b;
        2.0 * d.xyz().length().atan2(d.w.abs())
    }

    #[test]
    fn inertia_matches_hand_calculation() {
        let (m, inertia) = book_inertia();
        close(m, 0.49728, 1e-6, "book m");
        close(inertia.x, 9.2427776e-4, 1e-6, "book I1");
        close(inertia.y, 1.8440800e-3, 1e-6, "book I2");
        close(inertia.z, 2.7352058e-3, 1e-6, "book I3");
        let top = top_properties();
        close(top.m, 0.21740135, 1e-6, "top m");
        close(top.l, 0.03028092, 1e-6, "top l");
        close(top.weight_moment, 0.06455829, 1e-6, "top m g l");
        close(top.i1, 3.3750225e-4, 1e-6, "top I1");
        close(top.i3, 2.6509593e-4, 1e-6, "top I3");
        close(sleeping_threshold(), 35.21610, 1e-6, "wc");
        close(steady_precession(30f64.to_radians(), 60.0), 4.417377, 1e-6, "slow root");
    }

    #[test]
    fn book_conserves_energy_and_angular_momentum() {
        let mut book = Top::preset(1);
        let (mut energy, mut momentum): (f64, f64) = (0.0, 0.0);
        for _ in 0..60_000 {
            book.step(DT);
            energy = energy.max((book.energy() / book.energy0 - 1.0).abs());
            momentum = momentum.max((book.world_momentum() - book.world_momentum0).length() / book.world_momentum0.length());
        }
        assert!(energy < 1e-10, "energy {energy}");
        assert!(momentum < 1e-11, "angular momentum {momentum}");
    }

    #[test]
    fn top_conserves_energy_and_both_momenta() {
        let mut top = Top::preset(4);
        let (mut energy, mut lz, mut l3): (f64, f64, f64) = (0.0, 0.0, 0.0);
        for _ in 0..20_000 {
            top.step(DT);
            energy = energy.max((top.energy() / top.energy0 - 1.0).abs());
            lz = lz.max((top.momentum.dot(top.up()) / top.lz0 - 1.0).abs());
            l3 = l3.max((top.momentum.z / top.l3_0 - 1.0).abs());
        }
        assert!(energy < 1e-9, "energy {energy}");
        assert!(lz < 1e-11, "Lz {lz}");
        assert!(l3 < 1e-14, "L3 {l3}");
    }

    #[test]
    fn intermediate_axis_grows_at_the_predicted_rate() {
        let (_, inertia) = book_inertia();
        let w0 = TAU;
        let mut book = Top::book("test", 1, dvec3(1e-6, 1.0, 0.0) * w0);
        run(&mut book, 1.0);
        // A small w1 grows as cosh(lambda t), lambda = w2 sqrt((I2 - I1)(I3 - I2) / (I1 I3)).
        let lambda = w0 * ((inertia.y - inertia.x) * (inertia.z - inertia.y) / (inertia.x * inertia.z)).sqrt();
        close(book.angular_velocity().x / (1e-6 * w0), lambda.cosh(), 1e-6, "growth");
        close(lambda, 3.5776706, 1e-6, "lambda t");
    }

    #[test]
    fn flip_interval_matches_elliptic_integral() {
        for (w, interval) in [(dvec3(0.01, 1.0, 0.01), 3.1694577), (dvec3(0.02, 1.0, 0.0), 2.9564978)] {
            let mut book = Top::book("test", 1, w * TAU);
            let (_, inertia) = book_inertia();
            assert!((flip_interval(inertia, book.momentum) - interval).abs() < 1e-6, "formula {}", flip_interval(inertia, book.momentum));
            let mut intervals = Vec::new();
            let mut flips = 0;
            for _ in 0..60_000 {
                book.step(DT);
                if book.flips > flips {
                    flips = book.flips;
                    if flips >= 2 {
                        intervals.push(book.flip_interval.unwrap());
                    }
                }
            }
            if w.x == 0.01 {
                assert_eq!(book.flips, 19);
            }
            assert!(intervals.len() >= 17);
            for measured in intervals {
                assert!((measured - interval).abs() < 1e-6, "interval {measured}, expected {interval}");
            }
        }
    }

    #[test]
    fn only_the_intermediate_axis_flips() {
        for (preset, low, high) in [(2, 0.0, 2.5), (3, 0.0, 0.6), (1, 179.0, 180.0)] {
            let mut book = Top::preset(preset);
            let mut largest: f64 = 0.0;
            for _ in 0..60_000 {
                book.step(DT);
                largest = largest.max(book.axis(book.spin_axis).angle_between(book.world_momentum()).to_degrees());
            }
            assert!(largest > low && largest < high, "preset {preset}: {largest} deg");
        }
    }

    #[test]
    fn free_symmetric_top_circles_its_angular_momentum() {
        let props = top_properties();
        let mut top = Top::top("test", 0.3, 60.0, 0.0);
        top.weight_moment = 0.0;
        top.momentum = dvec3(0.004, 0.0, props.i3 * 60.0);
        let start = top.axis(2);
        let period = TAU * props.i1 / top.momentum.length();
        let steps = (period / DT).floor();
        for _ in 0..steps as usize {
            top.step(DT);
        }
        top.step(period - steps * DT);
        assert!(top.axis(2).angle_between(start) < 1e-12, "off by {}", top.axis(2).angle_between(start));
    }

    #[test]
    fn steady_precession_keeps_its_tilt() {
        let mut top = Top::preset(5);
        run(&mut top, 10.0);
        assert!((top.tilt() - 30f64.to_radians()).abs() < 1e-7, "tilt {}", top.tilt().to_degrees());
        assert!((top.azimuth - top.azimuth0 - 4.417377 * 10.0).abs() < 1e-5, "azimuth {}", top.azimuth - top.azimuth0);

        // Lying on its side the formula is exact: W = m g l / (I3 w3).
        let props = top_properties();
        let w = props.weight_moment / (props.i3 * 60.0);
        close(w, 4.058800, 1e-6, "horizontal precession");
        let mut top = Top::top("test", PI / 2.0, 60.0, w);
        run(&mut top, 10.0);
        assert!((top.tilt() - PI / 2.0).abs() < 1e-7, "tilt {}", top.tilt().to_degrees());
        assert!((top.azimuth - top.azimuth0 - w * 10.0).abs() < 1e-5, "azimuth {}", top.azimuth - top.azimuth0);
    }

    #[test]
    fn released_top_nods_between_the_predicted_limits() {
        let mut top = Top::preset(4);
        let mut smallest = f64::INFINITY;
        for _ in 0..20_000 {
            top.step(DT);
            smallest = smallest.min(top.tilt());
        }
        let limit = released_tilt_limit(30f64.to_radians(), 60.0).to_degrees();
        assert!((limit - 36.33638).abs() < 1e-4, "limit {limit}");
        assert!((smallest.to_degrees() - 30.0).abs() < 1e-4, "smallest {}", smallest.to_degrees());
        assert!((top.max_tilt.to_degrees() - limit).abs() < 1e-4, "largest {}", top.max_tilt.to_degrees());
    }

    #[test]
    fn sleeping_threshold_separates_sleeping_from_falling() {
        let mut sleeping = Top::preset(6);
        let mut falling = Top::preset(7);
        run(&mut sleeping, 20.0);
        run(&mut falling, 20.0);
        let (small, large) = (sleeping.max_tilt.to_degrees(), falling.max_tilt.to_degrees());
        assert!((small - 1.80861).abs() < 1e-4, "sleeping top reaches {small} deg");
        assert!((large - 51.73118).abs() < 1e-4, "slow top reaches {large} deg");
        // Started exactly upright it would reach acos(2 (w3 / wc)^2 - 1).
        assert!((large - (2.0f64 * 0.81 - 1.0).acos().to_degrees()).abs() < 0.05);
    }

    #[test]
    fn integrator_is_fourth_order() {
        let at = |dt: f64| {
            let mut top = Top::preset(4);
            for _ in 0..(2.0 / dt).round() as usize {
                top.step(dt);
            }
            top.q
        };
        let reference = at(1.0 / 4000.0);
        let (coarse, fine) = (orientation_error(at(1.0 / 250.0), reference), orientation_error(at(1.0 / 500.0), reference));
        let ratio = coarse / fine;
        assert!(ratio > 14.0 && ratio < 18.0, "error ratio {ratio} (errors {coarse:e} and {fine:e})");
    }
}
