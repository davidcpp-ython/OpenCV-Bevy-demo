//! # Bevy + OpenCV Real-time Camera Integration
//!
//! Demonstrates spawning Bevy sprites based on OpenCV corner detection from a live camera feed.
//!
//! ## Key Components
//! - [`CvCamera`] - OpenCV camera capture resource
//! - [`CornerDot`] - Marker component for spawned corner sprites
//! - [`setup`] - Initializes Bevy 2D camera and OpenCV camera
//! - [`detect_and_spawn_corners`] - Detects corners and spawns sprites in real-time

use anyhow::{Context, Result};
use bevy::prelude::*;
use opencv::{
    core::{self, Mat, Vector},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
};

/// OpenCV camera resource for frame capture
///
/// SAFETY: VideoCapture contains raw pointers and is not Send/Sync by default.
/// We mark it as such because we only access it from the main thread in Bevy systems.
struct CvCamera {
    cap: VideoCapture,
    frame_bgr: Mat,
    w: i32,
    h: i32,
}

// SAFETY: We only access CvCamera from the main Bevy thread
unsafe impl Send for CvCamera {}
unsafe impl Sync for CvCamera {}

impl bevy::prelude::Resource for CvCamera {}

impl CvCamera {
    /// Initialize camera with specified device index
    fn new(device: i32) -> Result<Self> {
        let cap = VideoCapture::new(device, videoio::CAP_ANY)
            .context("Failed to open camera device")?;

        if !cap.is_opened().context("Failed to check if camera is opened")? {
            anyhow::bail!("Camera is not opened");
        }

        // Get camera dimensions
        let w = cap
            .get(videoio::CAP_PROP_FRAME_WIDTH)
            .context("Failed to get frame width")? as i32;
        let h = cap
            .get(videoio::CAP_PROP_FRAME_HEIGHT)
            .context("Failed to get frame height")? as i32;

        info!("Camera initialized: {}x{}", w, h);

        Ok(Self {
            cap,
            frame_bgr: Mat::default(),
            w,
            h,
        })
    }
}

/// Marker component for corner detection sprites
#[derive(Component)]
struct CornerDot;

/// Marker component for the camera feed background sprite
#[derive(Component)]
struct CameraFeed;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (update_camera_feed, detect_and_spawn_corners).chain())
        .run();
}

/// Setup Bevy 2D camera and initialize OpenCV camera
fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // Spawn Bevy 2D camera (Required Components pattern in 0.16)
    commands.spawn(Camera2d);

    // Initialize OpenCV camera (device 0 = default camera)
    match CvCamera::new(0) {
        Ok(cam) => {
            let w = cam.w as u32;
            let h = cam.h as u32;

            // Create empty image for camera feed
            let image = Image::new_fill(
                bevy::render::render_resource::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                bevy::render::render_resource::TextureDimension::D2,
                &vec![0u8; (w * h * 4) as usize],
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                bevy::render::render_asset::RenderAssetUsages::MAIN_WORLD
                    | bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
            );

            let image_handle = images.add(image);

            // Spawn camera feed background sprite
            commands.spawn((
                Sprite::from_image(image_handle),
                Transform::from_xyz(0.0, 0.0, 0.0),
                CameraFeed,
            ));

            commands.insert_resource(cam);
            info!("OpenCV camera ready: {}x{}", w, h);
        }
        Err(e) => {
            error!("Failed to initialize camera: {:?}", e);
            std::process::exit(1);
        }
    }
}

/// Update camera feed texture with latest frame from OpenCV
fn update_camera_feed(
    mut cam: ResMut<CvCamera>,
    mut images: ResMut<Assets<Image>>,
    q_feed: Query<&Sprite, With<CameraFeed>>,
) {
    // Capture fresh frame from camera
    let CvCamera {
        ref mut cap,
        ref mut frame_bgr,
        w,
        h,
    } = *cam;

    let frame_read = cap.read(frame_bgr).unwrap_or(false);
    if !frame_read || frame_bgr.empty() {
        return;
    }

    // Convert BGR to RGBA for Bevy
    let mut frame_rgba = Mat::default();
    if imgproc::cvt_color(
        frame_bgr,
        &mut frame_rgba,
        imgproc::COLOR_BGR2RGBA,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .is_err()
    {
        return;
    }

    // Get the camera feed sprite's image handle
    if let Ok(sprite) = q_feed.get_single() {
        if let Some(image) = images.get_mut(&sprite.image) {
            // Copy OpenCV frame data to Bevy image
            if let Ok(data) = frame_rgba.data_bytes() {
                image.data.copy_from_slice(data);
            }
        }
    }
}

/// Detect corners using Shi-Tomasi algorithm and spawn sprites at corner locations
fn detect_and_spawn_corners(
    mut commands: Commands,
    mut cam: ResMut<CvCamera>,
    q_existing: Query<Entity, With<CornerDot>>,
) {
    // Clear old corner dots (simple approach; could also reuse entities)
    for entity in q_existing.iter() {
        commands.entity(entity).despawn();
    }

    // Capture fresh frame from camera
    let CvCamera {
        ref mut cap,
        ref mut frame_bgr,
        w,
        h,
    } = *cam;

    let frame_read = cap.read(frame_bgr).unwrap_or(false);
    if !frame_read || frame_bgr.empty() {
        return;
    }

    // Convert BGR to grayscale for corner detection
    let mut gray = Mat::default();
    if imgproc::cvt_color(
        frame_bgr,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .is_err()
    {
        return;
    }

    // Detect up to 200 strongest corners using Shi-Tomasi (goodFeaturesToTrack)
    let mut corners = Vector::<core::Point2f>::new();
    let detection_result = imgproc::good_features_to_track(
        &gray,
        &mut corners,
        200,              // max_corners
        0.01,             // quality_level
        10.0,             // min_distance between corners
        &core::no_array(), // mask
        3,                // block_size
        false,            // use_harris_detector
        0.04,             // k (Harris detector free parameter)
    );

    if detection_result.is_err() {
        return;
    }

    // Spawn a small colored sprite at each detected corner
    let w = w as f32;
    let h = h as f32;

    for i in 0..corners.len() {
        if let Ok(p) = corners.get(i) {
            // Map pixel coordinates (0..w, 0..h) to Bevy world coordinates (centered at origin)
            let x = p.x - w / 2.0;
            let y = -(p.y - h / 2.0); // Flip Y axis (OpenCV: +Y down, Bevy: +Y up)

            // Spawn sprite using Bevy 0.16 Required Components pattern
            commands.spawn((
                Sprite {
                    color: Color::srgb(1.0, 0.2, 0.2), // Red dots
                    custom_size: Some(Vec2::splat(4.0)), // 4x4 pixel dots
                    ..default()
                },
                Transform::from_xyz(x, y, 1.0),
                CornerDot,
            ));
        }
    }
}
