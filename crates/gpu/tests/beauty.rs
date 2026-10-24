//! CPU/GPU parity on the actual adapter, including independently adjusted two-face chains.

use filmcraft_frame::VideoFrame;
use filmcraft_gpu::GpuCompositor;
use filmcraft_render::Image;
use filmcraft_render::beauty::{BeautySettings, FaceRegion, RepairSpot};
use filmcraft_render::gpufx::FxOp;
use filmcraft_render::plan::LayerFx;

fn compositor() -> Option<GpuCompositor> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    eprintln!("Beauty parity adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(GpuCompositor::new(&device, &queue))
}

/// Colors/alpha are exact half floats so the GPU source draw adds no fixture quantization error.
fn picture(w: usize, h: usize) -> Image {
    let mut image = Image::filled(w, h, [0.375, 0.25, 0.1875, 1.0]);
    for y in 0..h {
        for x in 0..w {
            let u = (x % 16) as f32 / 16.0;
            let v = (y % 16) as f32 / 16.0;
            let alpha = match y % 13 {
                0 => 0.0,
                1..=3 => 0.25,
                4..=6 => 0.5,
                _ => 1.0,
            };
            let mut color = [0.375 + 0.125 * u, 0.25 + 0.0625 * v, 0.1875];
            if x % 23 == 7 && y % 19 == 8 {
                color[0] += 0.125;
            }
            if x % 29 == 17 {
                color = [0.0625, 0.03125, 0.015625];
            }
            if x % 31 == 13 {
                color = [1.25, 0.875, 0.5];
            }
            let i = (y * w + x) * 4;
            image.px[i..i + 4].copy_from_slice(&[color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha]);
        }
    }
    image
}

fn compare(compositor: &mut GpuCompositor, input: &Image, ops: Vec<FxOp>) -> Vec<f32> {
    assert!(ops.iter().all(FxOp::gpu_ok));
    let mut cpu = input.clone();
    for op in &ops {
        op.apply(&mut cpu);
    }
    let frame = VideoFrame::rgba_f32(input.w as u32, input.h as u32, input.px.clone());
    let layer = LayerFx { size: (input.w as u32, input.h as u32), decimation: 1, ops };
    let (_, _, gpu) = compositor.effect_image(&frame, &layer).expect("Beauty GPU working-image readback");
    assert_eq!(cpu.px.len(), gpu.len());
    let mut worst = 0.0_f32;
    let mut position = 0;
    for (index, (a, b)) in cpu.px.iter().zip(&gpu).enumerate() {
        assert!(b.is_finite(), "non-finite GPU sample at {index}");
        let error = (a - b).abs() / (1.0 + a.abs());
        if error > worst {
            worst = error;
            position = index;
        }
        if index % 4 == 3 {
            assert_eq!(a.to_bits(), b.to_bits(), "alpha at {index}");
        }
    }
    assert!(worst < 1e-4, "Beauty CPU/GPU error {worst:e} at {position}: CPU {}, GPU {}", cpu.px[position], gpu[position]);
    gpu
}

#[test]
fn beauty_all_controls_match_cpu_on_actual_gpu() {
    let Some(mut compositor) = compositor() else {
        eprintln!("no GPU adapter; skipping Beauty parity");
        return;
    };
    let face = FaceRegion::from_bounds([0.035, 0.045, 0.93, 0.91]);
    let default = BeautySettings { skin_only: false, ..BeautySettings::default() };
    let controls = [
        BeautySettings { smoothing: 1.0, detail: 0.0, radius: 0.5, ..default },
        BeautySettings { smoothing: 0.7, detail: 0.2, radius: 4.0, ..default },
        BeautySettings { smoothing: 1.0, detail: 1.0, radius: 32.0, ..default },
        BeautySettings { whitening: 1.0, ..default },
        BeautySettings { rosiness: 1.0, ..default },
        BeautySettings { blemish: 1.0, ..default },
        BeautySettings { nasolabial: 1.0, ..default },
        BeautySettings { slimming: 1.0, ..default },
        BeautySettings {
            smoothing: 0.8,
            whitening: 0.6,
            rosiness: 0.4,
            blemish: 0.8,
            nasolabial: 0.7,
            slimming: 0.6,
            detail: 0.5,
            radius: 5.0,
            skin_only: true,
        },
    ];
    for (w, h) in [(128, 96), (67, 41)] {
        let input = picture(w, h);
        for settings in controls {
            compare(&mut compositor, &input, vec![FxOp::Beauty { settings, face: face.clone() }]);
        }
    }
}

#[test]
fn beauty_two_faces_match_cpu_and_do_not_modify_the_unselected_person() {
    let Some(mut compositor) = compositor() else {
        eprintln!("no GPU adapter; skipping Beauty parity");
        return;
    };
    let input = picture(160, 96);
    let a = FaceRegion::from_bounds([0.03, 0.08, 0.42, 0.84]);
    let b = FaceRegion::from_bounds([0.55, 0.08, 0.42, 0.84]);
    let first = FxOp::Beauty { settings: BeautySettings { whitening: 0.8, blemish: 1.0, slimming: 0.4, ..BeautySettings::default() }, face: a };
    let gpu = compare(&mut compositor, &input, vec![first.clone()]);
    for y in 0..input.h {
        for x in 88..input.w {
            let index = (y * input.w + x) * 4;
            assert_eq!(&gpu[index..index + 4], &input.px[index..index + 4], "person B changed at {x},{y}");
        }
    }
    compare(
        &mut compositor,
        &input,
        vec![
            first,
            FxOp::Beauty {
                settings: BeautySettings { smoothing: 0.9, whitening: 0.1, rosiness: 0.3, nasolabial: 0.8, slimming: 1.0, ..BeautySettings::default() },
                face: b,
            },
        ],
    );
}

#[test]
fn manual_spots_and_invalid_geometry_require_cpu_fallback() {
    let mut face = FaceRegion::from_bounds([0.1, 0.1, 0.8, 0.8]);
    face.repair_spots.push(RepairSpot { center: [0.3, 0.6], radius: 0.02 });
    assert!(!FxOp::Beauty { settings: BeautySettings { blemish: 1.0, ..BeautySettings::default() }, face }.gpu_ok());
    let mut invalid = FaceRegion::from_bounds([0.1, 0.1, 0.8, 0.8]);
    invalid.landmarks[0][0] = f32::NAN;
    assert!(!FxOp::Beauty { settings: BeautySettings { whitening: 1.0, ..BeautySettings::default() }, face: invalid }.gpu_ok());
    let mut overlapping = FaceRegion::from_bounds([0.1, 0.1, 0.8, 0.8]);
    overlapping.exclusions.push([0.5, 0.1, 0.45, 0.8]);
    assert!(!FxOp::Beauty { settings: BeautySettings { whitening: 1.0, ..BeautySettings::default() }, face: overlapping }.gpu_ok());
}
