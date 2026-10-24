use filmcraft_render::Image;
use filmcraft_render::beauty::{BeautyError, BeautySettings, FaceRegion, RepairSpot, process};

fn face() -> FaceRegion {
    FaceRegion::from_bounds([0.1, 0.05, 0.8, 0.9])
}

fn settings() -> BeautySettings {
    BeautySettings { skin_only: false, ..BeautySettings::default() }
}

fn pixel(image: &Image, x: usize, y: usize) -> [f32; 4] {
    image.get(x, y)
}

fn put(image: &mut Image, x: usize, y: usize, c: [f32; 4]) {
    let i = (y * image.w + x) * 4;
    image.px[i..i + 4].copy_from_slice(&c);
}

#[test]
fn zero_controls_are_bit_exact_even_for_nonopaque_hdr() {
    let mut image = Image::filled(31, 17, [1.23456, 0.06234, -0.03, 0.314]);
    let original = image.clone();
    process(&mut image, settings(), None).unwrap();
    assert_eq!(image, original);
}

#[test]
fn active_controls_require_explicit_face() {
    let mut image = Image::filled(16, 16, [0.4, 0.25, 0.2, 1.0]);
    let original = image.clone();
    assert_eq!(process(&mut image, BeautySettings { whitening: 1.0, ..settings() }, None), Err(BeautyError::MissingFaceRegion));
    assert_eq!(image, original);
}

#[test]
fn smoothing_reduces_skin_noise_and_keeps_strong_edges() {
    let mut image = Image::filled(128, 128, [0.4, 0.24, 0.17, 1.0]);
    for y in 0..128 {
        for x in 0..128 {
            let base = if x < 64 { 0.4 } else { 0.03 };
            let n = if (x + y) % 2 == 0 { 0.012 } else { -0.012 };
            put(&mut image, x, y, [base + n, base * 0.6 + n, base * 0.4 + n, 1.0]);
        }
    }
    let original = image.clone();
    process(&mut image, BeautySettings { smoothing: 1.0, detail: 0.0, radius: 3.0, ..settings() }, Some(&face())).unwrap();
    let contrast_before = (pixel(&original, 42, 63)[0] - pixel(&original, 43, 63)[0]).abs();
    let contrast_after = (pixel(&image, 42, 63)[0] - pixel(&image, 43, 63)[0]).abs();
    assert!(contrast_after < contrast_before * 0.8, "noise reduction {contrast_before} -> {contrast_after}");
    assert!(pixel(&image, 63, 62)[0] > 0.35, "strong edge should not bleed into skin");
    assert!(pixel(&image, 64, 62)[0] < 0.06, "dark side should retain strong edge");
}

#[test]
fn detail_slider_preserves_more_texture() {
    let original = {
        let mut image = Image::filled(96, 96, [0.4, 0.25, 0.2, 1.0]);
        put(&mut image, 32, 48, [0.43, 0.28, 0.23, 1.0]);
        image
    };
    let mut soft = original.clone();
    let mut detailed = original.clone();
    process(&mut soft, BeautySettings { smoothing: 1.0, detail: 0.0, ..settings() }, Some(&face())).unwrap();
    process(&mut detailed, BeautySettings { smoothing: 1.0, detail: 1.0, ..settings() }, Some(&face())).unwrap();
    assert!(pixel(&detailed, 32, 48)[0] > pixel(&soft, 32, 48)[0] + 0.01);
}

#[test]
fn alpha_is_exact_and_hdr_headroom_is_not_clipped() {
    let mut image = Image::filled(64, 64, [1.3, 0.7, 0.4, 0.4]);
    put(&mut image, 24, 32, [0.0, 0.0, 0.0, 0.0]);
    let original = image.clone();
    process(&mut image, BeautySettings { whitening: 0.8, rosiness: 0.5, smoothing: 0.6, slimming: 0.5, ..settings() }, Some(&face())).unwrap();
    for (a, b) in image.px.as_chunks::<4>().0.iter().zip(original.px.as_chunks::<4>().0) {
        assert_eq!(a[3].to_bits(), b[3].to_bits());
    }
    assert!(pixel(&image, 22, 32)[0] > 1.3);
    assert_eq!(pixel(&image, 24, 32), [0.0; 4]);
    assert!(image.px.iter().all(|v| v.is_finite()));
}

#[test]
fn each_face_only_changes_its_own_region() {
    let original = Image::filled(160, 100, [0.4, 0.25, 0.18, 1.0]);
    let a = FaceRegion::from_bounds([0.03, 0.1, 0.4, 0.8]);
    let b = FaceRegion::from_bounds([0.57, 0.1, 0.4, 0.8]);
    let mut image = original.clone();
    process(&mut image, BeautySettings { whitening: 1.0, rosiness: 1.0, ..settings() }, Some(&a)).unwrap();
    assert!(pixel(&image, 40, 50)[0] > pixel(&original, 40, 50)[0]);
    for y in 0..100 {
        for x in 70..160 {
            assert_eq!(pixel(&image, x, y), pixel(&original, x, y));
        }
    }
    let after_a = image.clone();
    process(&mut image, BeautySettings { smoothing: 0.6, whitening: 0.5, slimming: 1.0, ..settings() }, Some(&b)).unwrap();
    for y in 0..100 {
        for x in 0..90 {
            assert_eq!(pixel(&image, x, y), pixel(&after_a, x, y));
        }
    }
}

#[test]
fn overlapping_peer_box_is_bit_exact_while_the_selected_face_is_retreated() {
    let original = Image::filled(160, 100, [0.4, 0.25, 0.18, 1.0]);
    let mut a = FaceRegion::from_bounds([0.1, 0.1, 0.6, 0.8]);
    a.exclusions.push([0.45, 0.15, 0.45, 0.8]);
    let mut image = original.clone();
    let active = BeautySettings { smoothing: 1.0, whitening: 1.0, rosiness: 1.0, blemish: 1.0, nasolabial: 1.0, slimming: 1.0, ..settings() };
    process(&mut image, active, Some(&a)).unwrap();
    assert!(pixel(&image, 48, 50)[0] > pixel(&original, 48, 50)[0]);
    for y in 15..95 {
        for x in 72..144 {
            assert_eq!(pixel(&image, x, y), pixel(&original, x, y), "peer face changed at {x},{y}");
        }
    }
}

#[test]
fn hostile_peer_exclusion_boxes_are_atomic_errors() {
    let active = BeautySettings { whitening: 1.0, ..settings() };
    for bounds in [[f32::NAN, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.5], [-0.1, 0.0, 0.5, 0.5], [0.8, 0.0, 0.5, 0.5], [0.0, 0.0, f32::MAX, 0.5]] {
        let mut selected = face();
        selected.exclusions.push(bounds);
        let mut image = Image::filled(2, 2, [0.4, 0.25, 0.18, 1.0]);
        let original = image.clone();
        assert_eq!(process(&mut image, active, Some(&selected)), Err(BeautyError::InvalidExclusion));
        assert_eq!(image, original);
    }
    let mut selected = face();
    selected.exclusions = vec![[0.3, 0.4, 0.2, 0.3]; 2];
    let mut image = Image::filled(2, 2, [0.4, 0.25, 0.18, 1.0]);
    assert_eq!(process(&mut image, active, Some(&selected)), Err(BeautyError::InvalidExclusion));
}

#[test]
fn auto_spot_treatment_softens_isolated_redness_without_removing_dark_moles() {
    let mut image = Image::filled(128, 128, [0.4, 0.24, 0.17, 1.0]);
    put(&mut image, 44, 68, [0.54, 0.21, 0.16, 1.0]);
    put(&mut image, 83, 68, [0.14, 0.09, 0.06, 1.0]);
    let mole = pixel(&image, 83, 68);
    process(&mut image, BeautySettings { blemish: 1.0, ..settings() }, Some(&face())).unwrap();
    assert!(pixel(&image, 44, 68)[0] < 0.49, "red spot was not softened");
    assert_eq!(pixel(&image, 83, 68), mole, "dark marks require manual selection");
}

#[test]
fn explicit_repair_softens_a_selected_dark_spot_only() {
    let mut image = Image::filled(128, 128, [0.4, 0.24, 0.17, 1.0]);
    put(&mut image, 44, 68, [0.14, 0.09, 0.06, 1.0]);
    put(&mut image, 83, 68, [0.14, 0.09, 0.06, 1.0]);
    let mut selected = face();
    selected.repair_spots.push(RepairSpot { center: [44.5 / 128.0, 68.5 / 128.0], radius: 0.025 });
    process(&mut image, BeautySettings { blemish: 1.0, ..settings() }, Some(&selected)).unwrap();
    assert!(pixel(&image, 44, 68)[0] > 0.35);
    assert_eq!(pixel(&image, 83, 68)[0], 0.14);
}

#[test]
fn nasolabial_control_lifts_a_landmark_localized_shadow() {
    let mut image = Image::filled(128, 128, [0.4, 0.24, 0.17, 1.0]);
    // Halfway between the manual left nostril and left mouth corner.
    put(&mut image, 53, 82, [0.29, 0.13, 0.06, 1.0]);
    let original = image.clone();
    process(&mut image, BeautySettings { nasolabial: 1.0, ..settings() }, Some(&face())).unwrap();
    assert!(pixel(&image, 53, 82)[0] > pixel(&original, 53, 82)[0] + 0.025);
    assert_eq!(pixel(&image, 39, 50), pixel(&original, 39, 50), "eye must remain untouched");
}

#[test]
fn slimming_moves_cheek_texture_inwards_without_touching_outside_box() {
    let mut image = Image::filled(128, 128, [0.4, 0.25, 0.18, 1.0]);
    for y in 0..128 {
        for x in 0..128 {
            put(&mut image, x, y, [x as f32 / 128.0, 0.25, 0.18, 1.0]);
        }
    }
    let original = image.clone();
    process(&mut image, BeautySettings { slimming: 1.0, ..settings() }, Some(&face())).unwrap();
    assert!(pixel(&image, 31, 76)[0] < pixel(&original, 31, 76)[0] - 0.005);
    assert!(pixel(&image, 96, 76)[0] > pixel(&original, 96, 76)[0] + 0.005);
    assert_eq!(pixel(&image, 3, 76), pixel(&original, 3, 76));
    assert_eq!(pixel(&image, 64, 48), pixel(&original, 64, 48));
}

#[test]
fn normalized_whitening_is_consistent_across_preview_resolutions() {
    let mut full = Image::filled(256, 256, [0.4, 0.25, 0.18, 1.0]);
    let mut half = Image::filled(128, 128, [0.4, 0.25, 0.18, 1.0]);
    let s = BeautySettings { whitening: 0.7, rosiness: 0.4, ..settings() };
    process(&mut full, s, Some(&face())).unwrap();
    process(&mut half, s, Some(&face())).unwrap();
    for k in 0..3 {
        assert!((pixel(&full, 88, 128)[k] - pixel(&half, 44, 64)[k]).abs() < 0.002);
    }
}

#[test]
fn skin_gate_supports_dark_and_light_skin_and_rejects_blue() {
    for color in [[0.055, 0.025, 0.012, 1.0], [0.7, 0.46, 0.34, 1.0]] {
        let mut image = Image::filled(64, 64, color);
        process(&mut image, BeautySettings { whitening: 1.0, ..BeautySettings::default() }, Some(&face())).unwrap();
        assert!(pixel(&image, 22, 32)[0] > color[0] + 0.02);
    }
    let mut blue = Image::filled(64, 64, [0.02, 0.05, 0.8, 1.0]);
    let original = blue.clone();
    process(&mut blue, BeautySettings { whitening: 1.0, ..BeautySettings::default() }, Some(&face())).unwrap();
    assert_eq!(blue, original);
}

#[test]
fn scaled_filter_radius_and_warp_agree_at_preview_resolution() {
    fn gradient(size: usize) -> Image {
        let mut image = Image::filled(size, size, [0.4, 0.25, 0.18, 1.0]);
        for y in 0..size {
            for x in 0..size {
                let u = (x as f32 + 0.5) / size as f32;
                let v = (y as f32 + 0.5) / size as f32;
                let n = (u * 40.0 + v * 30.0).sin() * 0.02;
                put(&mut image, x, y, [0.3 + u * 0.1 + n, 0.22 + v * 0.07 + n, 0.15 + n, 1.0]);
            }
        }
        image
    }
    let mut full = gradient(256);
    let mut half = gradient(128);
    let s = BeautySettings { smoothing: 0.8, slimming: 0.8, radius: 8.0, ..settings() };
    process(&mut full, s, Some(&face())).unwrap();
    process(&mut half, BeautySettings { radius: 4.0, ..s }, Some(&face())).unwrap();
    let full = full.downsample2();
    for y in 20..108 {
        for x in 20..108 {
            for k in 0..3 {
                assert!((pixel(&full, x, y)[k] - pixel(&half, x, y)[k]).abs() < 0.003, "scale disagreement at {x},{y},{k}");
            }
        }
    }
}

#[test]
fn extreme_finite_hdr_samples_cannot_create_nonfinite_output() {
    let active = BeautySettings { smoothing: 1.0, whitening: 1.0, rosiness: 1.0, blemish: 1.0, nasolabial: 1.0, slimming: 1.0, ..settings() };
    for value in [f32::MAX, -f32::MAX, 1e30, -1e30] {
        let mut image = Image::filled(8, 8, [value, value * 0.5, value * 0.1, 0.01]);
        process(&mut image, active, Some(&face())).unwrap();
        assert!(image.px.iter().all(|v| v.is_finite()));
        assert!(image.px.as_chunks::<4>().0.iter().all(|p| p[3] == 0.01));
    }
}

#[test]
fn hostile_images_geometry_and_settings_never_panic_or_partially_modify() {
    let active = BeautySettings { smoothing: 1.0, whitening: 1.0, blemish: 1.0, slimming: 1.0, ..settings() };
    let inputs = [
        Image { w: 0, h: 1, px: vec![] },
        Image { w: usize::MAX, h: 2, px: vec![] },
        Image { w: 10, h: 10, px: vec![0.0; 12] },
        Image { w: 1, h: 1, px: vec![f32::NAN, 0.0, 0.0, 1.0] },
        Image { w: 1, h: 1, px: vec![0.1, 0.1, 0.1, -0.5] },
    ];
    for mut image in inputs {
        let before = image.px.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| process(&mut image, active, Some(&face()))));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
        assert_eq!(image.px.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
    }
    for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY, -1.0, f32::MAX] {
        let mut selected = face();
        selected.landmarks[0][0] = bad;
        let mut image = Image::filled(1, 1, [0.4, 0.25, 0.18, 1.0]);
        assert!(process(&mut image, active, Some(&selected)).is_err());
        let s = BeautySettings { smoothing: bad, whitening: bad, radius: bad, detail: bad, ..settings() };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| process(&mut image, s, Some(&face())))).is_ok());
    }
}

#[test]
fn repair_spot_count_and_invalid_radius_return_errors() {
    let mut image = Image::filled(16, 16, [0.4, 0.25, 0.18, 1.0]);
    let mut selected = face();
    let active = BeautySettings { blemish: 1.0, ..settings() };
    for radius in [f32::NAN, f32::INFINITY, -1.0, 0.0, 0.21] {
        selected.repair_spots = vec![RepairSpot { center: [0.3, 0.5], radius }];
        assert_eq!(process(&mut image, active, Some(&selected)), Err(BeautyError::InvalidRepairSpot));
    }
    selected.repair_spots = vec![RepairSpot { center: [0.3, 0.5], radius: 0.02 }; 65];
    assert_eq!(process(&mut image, active, Some(&selected)), Err(BeautyError::InvalidRepairSpot));
}

#[test]
fn malformed_bounds_are_atomic_errors_even_for_subpixel_faces() {
    let active = BeautySettings { whitening: 1.0, ..settings() };
    for bounds in [
        [f32::NAN, 0.0, 0.5, 0.5],
        [0.0, f32::INFINITY, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.5],
        [-0.1, 0.1, 0.5, 0.5],
        [0.8, 0.0, 0.5, 0.5],
        [0.0, 0.0, f32::MAX, 0.5],
    ] {
        let mut image = Image::filled(2, 2, [0.4, 0.25, 0.18, 1.0]);
        let original = image.clone();
        let selected = FaceRegion::from_bounds(bounds);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| process(&mut image, active, Some(&selected))));
        assert_eq!(result.unwrap(), Err(BeautyError::InvalidFaceGeometry));
        assert_eq!(image, original);
    }
    let mut image = Image::filled(2, 2, [0.4, 0.25, 0.18, 1.0]);
    let selected = FaceRegion::from_bounds([0.5, 0.5, 1e-20, 1e-20]);
    assert!(process(&mut image, active, Some(&selected)).is_ok());
}

#[test]
#[ignore = "manual CPU timing, depends on host load"]
fn perf_two_faces_1080p() {
    let mut image = Image::filled(1920, 1080, [0.4, 0.25, 0.18, 1.0]);
    let a = FaceRegion::from_bounds([0.08, 0.15, 0.32, 0.7]);
    let b = FaceRegion::from_bounds([0.60, 0.15, 0.32, 0.7]);
    let active = BeautySettings { smoothing: 0.6, whitening: 0.3, rosiness: 0.2, blemish: 0.5, nasolabial: 0.5, slimming: 0.4, ..BeautySettings::default() };
    let start = std::time::Instant::now();
    process(&mut image, active, Some(&a)).unwrap();
    process(&mut image, active, Some(&b)).unwrap();
    eprintln!("beauty two faces 1080p CPU: {:?}", start.elapsed());
}
