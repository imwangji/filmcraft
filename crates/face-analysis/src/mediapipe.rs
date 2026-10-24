//! Pure-Rust CPU execution of Google's pinned legacy BlazeFace + Face Mesh models.
//!
//! Graphs and tensors run under an unwind boundary. Weight bytes are size and SHA-256
//! verified before any third-party parser sees them. No TFLite native library is used.
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use sha2::{Digest, Sha256};
use tract_tflite::prelude::*;

use crate::{AnalysisOptions, FaceAnalyzer, FaceBounds, FaceDetection, FaceError, Landmark, RgbImage, models};

type Network = std::sync::Arc<TypedRunnableModel>;
const DETECTOR_SIDE: usize = 128;
const MESH_SIDE: usize = 192;
const ANCHOR_COUNT: usize = 896;
const LANDMARK_COUNT: usize = 468;

pub struct MediaPipeAnalyzer {
    detector: Network,
    landmark: Network,
}

impl MediaPipeAnalyzer {
    /// Load only the pinned official files listed in [`models::FILES`].
    /// Neither construction nor inference accesses the network.
    pub fn from_dir(path: &Path) -> Result<Self, FaceError> {
        guarded(|| {
            let detector = load(path, &models::FILES[0])?;
            let landmark = load(path, &models::FILES[1])?;
            Ok(Self { detector, landmark })
        })
    }

    fn analyze_inner(&self, image: RgbImage<'_>, options: AnalysisOptions) -> Result<Vec<FaceDetection>, FaceError> {
        image.validate()?;
        options.validate()?;
        let width = image.width as f32;
        let height = image.height as f32;
        let side = width.max(height);
        let detector_crop = Crop { cx: width * 0.5, cy: height * 0.5, side, angle: 0.0 };
        let detector_input = resample(image, detector_crop, DETECTOR_SIDE, true)?;
        let raw = run(&self.detector, detector_input, DETECTOR_SIDE)?;
        let boxes = output(&raw, ANCHOR_COUNT * 16)?;
        let scores = output(&raw, ANCHOR_COUNT)?;
        // Keep a few extra proposals in case the mesh rejects a false positive.
        let proposals = decode_detections(&boxes, &scores, options.min_confidence, (options.max_faces * 4).min(32))?;
        let mut faces = Vec::with_capacity(options.max_faces);
        for detection in proposals {
            let cx = (detection.bounds.left + detection.bounds.right) * 0.5;
            let cy = (detection.bounds.top + detection.bounds.bottom) * 0.5;
            let crop = Crop {
                cx: (cx - 0.5) * side + width * 0.5,
                cy: (cy - 0.5) * side + height * 0.5,
                side: (detection.bounds.right - detection.bounds.left).max(detection.bounds.bottom - detection.bounds.top) * side * 1.5,
                angle: detection.angle,
            };
            let input = resample(image, crop, MESH_SIDE, false)?;
            let raw = run(&self.landmark, input, MESH_SIDE)?;
            let coordinates = output(&raw, LANDMARK_COUNT * 3)?;
            let presence = output(&raw, 1)?;
            let presence = presence.first().copied().filter(|x| x.is_finite()).ok_or_else(|| FaceError::Model("non-finite mesh confidence".into()))?;
            let confidence = sigmoid(presence).min(detection.confidence);
            if confidence < options.min_confidence {
                continue;
            }
            let landmarks = decode_landmarks(&coordinates, crop, width, height)?;
            let bounds = landmark_bounds(&landmarks)?;
            faces.push(FaceDetection { bounds, landmarks, confidence });
            if faces.len() == options.max_faces {
                break;
            }
        }
        Ok(faces)
    }
}

impl FaceAnalyzer for MediaPipeAnalyzer {
    fn analyze(&self, image: RgbImage<'_>, options: AnalysisOptions) -> Result<Vec<FaceDetection>, FaceError> {
        guarded(|| self.analyze_inner(image, options))
    }
}

fn guarded<T>(f: impl FnOnce() -> Result<T, FaceError>) -> Result<T, FaceError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| FaceError::Model("face inference failed safely; no analysis was saved".into()))?
}

fn load(directory: &Path, spec: &models::ModelFile) -> Result<Network, FaceError> {
    let path = directory.join(spec.name);
    let file = std::fs::File::open(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            FaceError::NotInstalled(path.display().to_string())
        } else {
            FaceError::Io(format!("{}: {e}", path.display()))
        }
    })?;
    let capacity = usize::try_from(spec.size).map_err(|_| FaceError::Model("weight size exceeds this platform".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(spec.size.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| FaceError::Io(e.to_string()))?;
    if bytes.len() != capacity || format!("{:x}", Sha256::digest(&bytes)) != spec.sha256 {
        return Err(FaceError::Model(format!("{} failed size/SHA-256 verification; reinstall the official model", spec.name)));
    }
    Ok(crate::fold::import(bytes)?)
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|e| FaceError::Model(format!("{}: {e:#}", spec.name)))
}

fn run(network: &Network, data: Vec<f32>, side: usize) -> Result<TVec<TValue>, FaceError> {
    let input = Tensor::from_shape(&[1, side, side, 3], &data).map_err(|e| FaceError::Model(e.to_string()))?;
    network.run(tvec![input.into()]).map_err(|e| FaceError::Model(e.to_string()))
}

fn output(raw: &[TValue], length: usize) -> Result<Vec<f32>, FaceError> {
    for value in raw {
        if value.len() == length {
            let value = value.cast_to::<f32>().map_err(|e| FaceError::Model(e.to_string()))?;
            let array = value.to_plain_array_view::<f32>().map_err(|e| FaceError::Model(e.to_string()))?;
            let result: Vec<f32> = array.iter().copied().collect();
            if result.iter().any(|n| !n.is_finite()) {
                return Err(FaceError::Model("non-finite network output".into()));
            }
            return Ok(result);
        }
    }
    Err(FaceError::Model(format!("expected a {length}-element output tensor")))
}

#[derive(Clone, Copy, Debug)]
struct Crop {
    cx: f32,
    cy: f32,
    side: f32,
    angle: f32,
}

impl Crop {
    fn project(self, x: f32, y: f32) -> (f32, f32) {
        let (s, c) = self.angle.sin_cos();
        (self.cx + self.side * (c * (x - 0.5) - s * (y - 0.5)), self.cy + self.side * (s * (x - 0.5) + c * (y - 0.5)))
    }
}

fn resample(image: RgbImage<'_>, crop: Crop, side: usize, signed: bool) -> Result<Vec<f32>, FaceError> {
    if !crop.cx.is_finite()
        || !crop.cy.is_finite()
        || !crop.side.is_finite()
        || !crop.angle.is_finite()
        || crop.side <= 0.0
        || !matches!(side, DETECTOR_SIDE | MESH_SIDE)
    {
        return Err(FaceError::Model("invalid face crop".into()));
    }
    let mut data = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            let (sx, sy) = crop.project((x as f32 + 0.5) / side as f32, (y as f32 + 0.5) / side as f32);
            let pixel = sample(image, sx - 0.5, sy - 0.5);
            for channel in pixel {
                let normalized = channel / 255.0;
                data.push(if signed { normalized * 2.0 - 1.0 } else { normalized });
            }
        }
    }
    Ok(data)
}

fn sample(image: RgbImage<'_>, x: f32, y: f32) -> [f32; 3] {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x > image.width.saturating_sub(1) as f32 || y > image.height.saturating_sub(1) as f32 {
        return [0.0; 3];
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0.saturating_add(1).min(image.width.saturating_sub(1));
    let y1 = y0.saturating_add(1).min(image.height.saturating_sub(1));
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut result = [0.0; 3];
    for (channel, value) in result.iter_mut().enumerate() {
        let at = |px: u32, py: u32| {
            let offset = (u64::from(py) * u64::from(image.width) + u64::from(px)) * u64::from(image.channels) + channel as u64;
            usize::try_from(offset).ok().and_then(|n| image.pixels.get(n)).copied().unwrap_or(0) as f32
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        *value = top * (1.0 - fy) + bottom * fy;
    }
    result
}

#[derive(Debug, Clone)]
struct Proposal {
    bounds: FaceBounds,
    confidence: f32,
    angle: f32,
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-100.0, 100.0)).exp())
}

fn anchors() -> Vec<(f32, f32)> {
    // BlazeFace's fixed-size SSD anchors: 2 per stride-8 cell, 6 per stride-16 cell.
    // This is generated mathematical grid geometry, not an embedded/traced visual asset.
    let mut anchors = Vec::with_capacity(ANCHOR_COUNT);
    for (grid, repetitions) in [(16, 2), (8, 6)] {
        for y in 0..grid {
            for x in 0..grid {
                for _ in 0..repetitions {
                    anchors.push(((x as f32 + 0.5) / grid as f32, (y as f32 + 0.5) / grid as f32));
                }
            }
        }
    }
    anchors
}

fn decode_detections(boxes: &[f32], scores: &[f32], minimum: f32, maximum: usize) -> Result<Vec<Proposal>, FaceError> {
    if boxes.len() != ANCHOR_COUNT * 16 || scores.len() != ANCHOR_COUNT || boxes.iter().chain(scores).any(|x| !x.is_finite()) {
        return Err(FaceError::Model("invalid detector tensor lengths or values".into()));
    }
    let mut proposals = Vec::new();
    for ((values, score), anchor) in boxes.as_chunks::<16>().0.iter().zip(scores.iter()).zip(anchors()) {
        let confidence = sigmoid(*score);
        if confidence < minimum {
            continue;
        }
        let value = |n: usize| values.get(n).copied().unwrap_or(0.0) / DETECTOR_SIDE as f32;
        let width = value(2);
        let height = value(3);
        if width <= 0.0 || height <= 0.0 || width > 2.0 || height > 2.0 {
            continue;
        }
        let cx = value(0) + anchor.0;
        let cy = value(1) + anchor.1;
        if !(-0.5..=1.5).contains(&cx) || !(-0.5..=1.5).contains(&cy) {
            continue;
        }
        let bounds = FaceBounds { left: cx - width * 0.5, top: cy - height * 0.5, right: cx + width * 0.5, bottom: cy + height * 0.5 };
        let eye_dx = value(6) - value(4);
        let eye_dy = value(7) - value(5);
        let angle = eye_dy.atan2(eye_dx);
        if eye_dx.hypot(eye_dy) < 0.005 {
            continue;
        }
        proposals.push(Proposal { bounds, confidence, angle });
    }
    proposals.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut selected: Vec<Proposal> = Vec::with_capacity(maximum);
    for proposal in proposals {
        if selected.iter().any(|p| overlap(proposal.bounds, p.bounds) > 0.3) {
            continue;
        }
        selected.push(proposal);
        if selected.len() == maximum {
            break;
        }
    }
    Ok(selected)
}

fn overlap(a: FaceBounds, b: FaceBounds) -> f32 {
    let intersection = (a.right.min(b.right) - a.left.max(b.left)).max(0.0) * (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0);
    let area = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection;
    if area > 0.0 { intersection / area } else { 0.0 }
}

fn decode_landmarks(coordinates: &[f32], crop: Crop, width: f32, height: f32) -> Result<Vec<Landmark>, FaceError> {
    if coordinates.len() != LANDMARK_COUNT * 3 || coordinates.iter().any(|n| !n.is_finite() || n.abs() > 4096.0) {
        return Err(FaceError::Model("invalid mesh tensor".into()));
    }
    let mut points = Vec::with_capacity(LANDMARK_COUNT);
    for xyz in coordinates.as_chunks::<3>().0 {
        let x = xyz.first().copied().ok_or_else(|| FaceError::Model("missing landmark x".into()))?;
        let y = xyz.get(1).copied().ok_or_else(|| FaceError::Model("missing landmark y".into()))?;
        let z = xyz.get(2).copied().ok_or_else(|| FaceError::Model("missing landmark z".into()))?;
        let (x, y) = crop.project(x / MESH_SIDE as f32, y / MESH_SIDE as f32);
        points.push(Landmark { x: (x / width).clamp(0.0, 1.0), y: (y / height).clamp(0.0, 1.0), z: z / MESH_SIDE as f32 * crop.side / width });
    }
    Ok(points)
}

fn landmark_bounds(landmarks: &[Landmark]) -> Result<FaceBounds, FaceError> {
    let mut bounds = FaceBounds { left: 1.0, top: 1.0, right: 0.0, bottom: 0.0 };
    for p in landmarks {
        bounds.left = bounds.left.min(p.x);
        bounds.right = bounds.right.max(p.x);
        bounds.top = bounds.top.min(p.y);
        bounds.bottom = bounds.bottom.max(p.y);
    }
    if bounds.left >= bounds.right || bounds.top >= bounds.bottom {
        return Err(FaceError::Model("degenerate face landmarks".into()));
    }
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires official local model weights and portrait test image"]
    fn official_models_detect_portrait_and_two_faces() {
        let path = std::env::var("FILMCRAFT_FACE_MODEL_DIR").expect("set FILMCRAFT_FACE_MODEL_DIR");
        let analyzer = MediaPipeAnalyzer::from_dir(Path::new(&path)).expect("load official models");
        let picture = image::open(Path::new(&path).join("portrait.jpg")).expect("portrait fixture").to_rgb8();
        let one = analyzer
            .analyze(RgbImage { width: picture.width(), height: picture.height(), channels: 3, pixels: picture.as_raw() }, AnalysisOptions::default())
            .expect("analyze portrait");
        println!(
            "portrait {}x{} result {:?}",
            picture.width(),
            picture.height(),
            one.iter().map(|f| (f.bounds, f.confidence, f.landmarks.get(1))).collect::<Vec<_>>()
        );
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].landmarks.len(), 468);
        let square = image::imageops::resize(&picture, 256, 320, image::imageops::FilterType::Triangle);
        let mut two = image::RgbImage::new(512, 320);
        image::imageops::replace(&mut two, &square, 0, 0);
        image::imageops::replace(&mut two, &square, 256, 0);
        let pair =
            analyzer.analyze(RgbImage { width: 512, height: 320, channels: 3, pixels: two.as_raw() }, AnalysisOptions::default()).expect("analyze two faces");
        println!("pair result {:?}", pair.iter().map(|f| (f.bounds, f.confidence, f.landmarks.get(1))).collect::<Vec<_>>());
        assert_eq!(pair.len(), 2);
    }
}

#[cfg(test)]
mod safety_tests {
    use super::*;

    #[test]
    fn anchors_decode_and_suppress_duplicate_faces() {
        assert_eq!(anchors().len(), ANCHOR_COUNT);
        let mut boxes = vec![0.0f32; ANCHOR_COUNT * 16];
        let mut scores = vec![-100.0f32; ANCHOR_COUNT];
        for (index, center_x) in [(0, 0.25), (1, 0.25), (2, 0.75)] {
            let anchor = anchors()[index];
            let row = &mut boxes[index * 16..(index + 1) * 16];
            row[0] = (center_x - anchor.0) * 128.0;
            row[1] = (0.5 - anchor.1) * 128.0;
            row[2] = 32.0;
            row[3] = 48.0;
            row[4] = -8.0;
            row[6] = 8.0;
            scores[index] = 5.0 - index as f32 * 0.1;
        }
        let proposals = decode_detections(&boxes, &scores, 0.5, 2).unwrap();
        assert_eq!(proposals.len(), 2);
        assert!((proposals[0].bounds.left - 0.125).abs() < 1e-5);
        assert!((proposals[1].bounds.left - 0.625).abs() < 1e-5);
        boxes[0] = f32::NAN;
        assert!(decode_detections(&boxes, &scores, 0.5, 2).is_err());
        assert!(decode_detections(&[], &[], 0.5, 2).is_err());
    }

    #[test]
    fn crop_rotation_and_mesh_backprojection_match_source_coordinates() {
        let crop = Crop { cx: 50.0, cy: 40.0, side: 20.0, angle: std::f32::consts::FRAC_PI_2 };
        let mut coordinates = vec![0.0f32; LANDMARK_COUNT * 3];
        for p in coordinates.as_chunks_mut::<3>().0 {
            p.copy_from_slice(&[192.0, 96.0, -19.2]);
        }
        let landmarks = decode_landmarks(&coordinates, crop, 100.0, 80.0).unwrap();
        assert!((landmarks[0].x - 0.5).abs() < 1e-5);
        assert!((landmarks[0].y - 0.625).abs() < 1e-5);
        assert!((landmarks[0].z + 0.02).abs() < 1e-5);
        coordinates[20] = f32::INFINITY;
        assert!(decode_landmarks(&coordinates, crop, 100.0, 80.0).is_err());
    }

    #[test]
    fn corrupt_and_missing_weights_fail_before_the_graph_parser() {
        let directory = std::env::temp_dir().join(format!("filmcraft-face-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for bytes in [vec![], vec![0xff; 100], vec![0; models::FILES[0].size as usize], vec![0; models::FILES[0].size as usize + 100]] {
            std::fs::write(directory.join(models::FILES[0].name), bytes).unwrap();
            assert!(matches!(MediaPipeAnalyzer::from_dir(&directory), Err(FaceError::Model(_))));
        }
        std::fs::remove_file(directory.join(models::FILES[0].name)).unwrap();
        assert!(matches!(MediaPipeAnalyzer::from_dir(&directory), Err(FaceError::NotInstalled(_))));
        std::fs::remove_dir(&directory).unwrap();
    }
}
