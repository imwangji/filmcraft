//! Original, deterministic CPU beauty reference. All geometry is normalized to the frame.
//!
//! This is a local image filter, not a face/medical classifier: automatic spot treatment only
//! attenuates isolated redness. Explicit repair spots also treat dark marks. Nasolabial treatment
//! softens shadows beside the supplied nose/mouth landmarks. Tracking and identity live upstream.

use std::fmt;

use rayon::prelude::*;

use crate::Image;

pub const EYE_LEFT: usize = 0;
pub const EYE_RIGHT: usize = 1;
pub const NOSE_TIP: usize = 2;
pub const MOUTH_LEFT: usize = 3;
pub const MOUTH_RIGHT: usize = 4;
pub const CHIN: usize = 5;
pub const CHEEK_LEFT: usize = 6;
pub const CHEEK_RIGHT: usize = 7;
pub const NOSE_LEFT: usize = 8;
pub const NOSE_RIGHT: usize = 9;

/// Bounded independent controls. Amounts and detail are 0..1; radius is working-image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeautySettings {
    pub smoothing: f32,
    pub whitening: f32,
    pub rosiness: f32,
    pub blemish: f32,
    pub nasolabial: f32,
    pub slimming: f32,
    pub detail: f32,
    pub radius: f32,
    pub skin_only: bool,
}

impl Default for BeautySettings {
    fn default() -> Self {
        Self { smoothing: 0.0, whitening: 0.0, rosiness: 0.0, blemish: 0.0, nasolabial: 0.0, slimming: 0.0, detail: 0.5, radius: 4.0, skin_only: true }
    }
}

impl BeautySettings {
    /// Rendering remains safe even when a damaged project bypasses command validation.
    pub fn sanitized(self) -> Self {
        fn amount(v: f32) -> f32 {
            if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
        }
        Self {
            smoothing: amount(self.smoothing),
            whitening: amount(self.whitening),
            rosiness: amount(self.rosiness),
            blemish: amount(self.blemish),
            nasolabial: amount(self.nasolabial),
            slimming: amount(self.slimming),
            detail: if self.detail.is_finite() { amount(self.detail) } else { 1.0 },
            radius: if self.radius.is_finite() { self.radius.clamp(0.5, 32.0) } else { 4.0 },
            skin_only: self.skin_only,
        }
    }

    pub fn is_identity(self) -> bool {
        let s = self.sanitized();
        s.smoothing == 0.0 && s.whitening == 0.0 && s.rosiness == 0.0 && s.blemish == 0.0 && s.nasolabial == 0.0 && s.slimming == 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepairSpot {
    pub center: [f32; 2],
    /// Radius relative to face-box width, in (0, .2].
    pub radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceRegion {
    /// [left, top, width, height], normalized to the image (not the selected face).
    pub bounds: [f32; 4],
    /// Eyes, nose tip, mouth corners, chin, cheeks, nostrils; see the named index constants.
    pub landmarks: [[f32; 2]; 10],
    pub repair_spots: Vec<RepairSpot>,
    /// At most one peer face box, xywh normalized to the frame. Its pixels remain untouched.
    /// This conservative protection does not infer which face is in front during an overlap.
    pub exclusions: Vec<[f32; 4]>,
}

impl FaceRegion {
    /// Explicit manual estimate when no landmark model is available. Does not detect a face.
    pub fn from_bounds(bounds: [f32; 4]) -> Self {
        let [x, y, w, h] = bounds;
        let point = |u, v| [x + w * u, y + h * v];
        Self {
            bounds,
            landmarks: [
                point(0.31, 0.36),
                point(0.69, 0.36),
                point(0.5, 0.56),
                point(0.36, 0.73),
                point(0.64, 0.73),
                point(0.5, 0.96),
                point(0.12, 0.60),
                point(0.88, 0.60),
                point(0.43, 0.59),
                point(0.57, 0.59),
            ],
            repair_spots: Vec::new(),
            exclusions: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), BeautyError> {
        let [x, y, w, h] = self.bounds;
        if !self.bounds.iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 || x < 0.0 || y < 0.0 || x + w > 1.000_001 || y + h > 1.000_001 {
            return Err(BeautyError::InvalidFaceGeometry);
        }
        if !self.landmarks.iter().flatten().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            return Err(BeautyError::InvalidFaceGeometry);
        }
        // Validated points must also belong to this face: a corrupt landmark cannot affect a peer.
        if self.landmarks.iter().any(|p| p[0] < x || p[0] > x + w || p[1] < y || p[1] > y + h) {
            return Err(BeautyError::InvalidFaceGeometry);
        }
        if self.repair_spots.len() > 64
            || self.repair_spots.iter().any(|p| {
                !p.radius.is_finite()
                    || p.radius <= 0.0
                    || p.radius > 0.2
                    || !p.center.iter().all(|v| v.is_finite())
                    || p.center[0] < x
                    || p.center[0] > x + w
                    || p.center[1] < y
                    || p.center[1] > y + h
            })
        {
            return Err(BeautyError::InvalidRepairSpot);
        }
        if self.exclusions.len() > 1
            || self.exclusions.iter().any(|bounds| {
                let [x, y, w, h] = *bounds;
                !bounds.iter().all(|v| v.is_finite()) || x < 0.0 || y < 0.0 || w <= 0.0 || h <= 0.0 || x + w > 1.000_001 || y + h > 1.000_001
            })
        {
            return Err(BeautyError::InvalidExclusion);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeautyError {
    InvalidImage,
    NonFinitePixel,
    MissingFaceRegion,
    InvalidFaceGeometry,
    InvalidRepairSpot,
    InvalidExclusion,
    AllocationFailed,
}

impl fmt::Display for BeautyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidImage => "beauty requires a nonempty RGBA image of at most 33,554,432 pixels with matching storage",
            Self::NonFinitePixel => "beauty input contains non-finite pixels or alpha outside 0..1",
            Self::MissingFaceRegion => "beauty requires an explicitly selected face region",
            Self::InvalidFaceGeometry => "beauty face bounds and landmarks must be finite and inside the image and selected face",
            Self::InvalidRepairSpot => "beauty accepts at most 64 finite repair spots inside the selected face, radius in (0, 0.2]",
            Self::InvalidExclusion => "beauty accepts at most one finite peer exclusion box wholly inside the image",
            Self::AllocationFailed => "beauty could not allocate its face-region working buffer",
        })
    }
}

impl std::error::Error for BeautyError {}

/// Validate once before the bounded hot loops. Errors leave the image completely unchanged.
pub fn process(image: &mut Image, settings: BeautySettings, face: Option<&FaceRegion>) -> Result<(), BeautyError> {
    let settings = settings.sanitized();
    if settings.is_identity() {
        return Ok(());
    }
    let count = image.w.checked_mul(image.h).filter(|n| *n > 0 && *n <= 33_554_432).ok_or(BeautyError::InvalidImage)?;
    let len = count.checked_mul(4).ok_or(BeautyError::InvalidImage)?;
    if image.px.len() != len || image.w > 32_768 || image.h > 32_768 {
        return Err(BeautyError::InvalidImage);
    }
    let face = face.ok_or(BeautyError::MissingFaceRegion)?;
    face.validate()?;
    if image.px.as_chunks::<4>().0.iter().any(|p| !p.iter().all(|v| v.is_finite()) || !(0.0..=1.0).contains(&p[3])) {
        return Err(BeautyError::NonFinitePixel);
    }
    let [left, top, width, height] = face.bounds;
    let x0 = ((left * image.w as f32).floor() as usize).min(image.w);
    let y0 = ((top * image.h as f32).floor() as usize).min(image.h);
    let x1 = (((left + width) * image.w as f32).ceil() as usize).min(image.w);
    let y1 = (((top + height) * image.h as f32).ceil() as usize).min(image.h);
    let (rw, rh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if rw == 0 || rh == 0 {
        return Ok(());
    }
    let capacity = rw.checked_mul(rh).ok_or(BeautyError::InvalidImage)?;
    let mut original = Vec::new();
    original.try_reserve_exact(capacity).map_err(|_| BeautyError::AllocationFailed)?;
    for y in y0..y1 {
        // Dimensions and storage length were validated above; loop bounds prove this slice fits.
        let start = (y * image.w + x0) * 4;
        let end = start + rw * 4;
        original.extend(image.px[start..end].as_chunks::<4>().0.iter().copied());
    }
    let source = Source { pixels: &original, w: rw, h: rh, x0, y0 };
    let geometry = Geometry::new(face, image.w as f32, image.h as f32);
    let stride = image.w * 4;
    image.px.par_chunks_mut(stride).enumerate().skip(y0).take(rh).for_each(|(y, row)| {
        for x in x0..x1 {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let mask = geometry.mask(point);
            if mask <= 0.0 {
                continue;
            }
            let original = source.sample(point);
            let alpha = original[3];
            if alpha <= 1e-6 {
                continue;
            }
            let sample_point = geometry.warp(point, settings.slimming, mask);
            let Some(color) = straight(source.sample(sample_point)) else { continue };
            let skin = if settings.skin_only { skin_weight(color) } else { 1.0 };
            let area = mask * geometry.feature_weight(point) * skin;
            if area == 0.0 && sample_point == point {
                continue;
            }
            let mut result = color;
            if area > 0.0 {
                let needs_local = settings.smoothing > 0.0 || settings.blemish > 0.0 || settings.nasolabial > 0.0;
                if needs_local {
                    if settings.smoothing > 0.0 {
                        let filtered = source.bilateral(sample_point, color, settings.radius);
                        let smooth = settings.smoothing * (1.0 - 0.8 * settings.detail);
                        result = mix(result, filtered, smooth * area);
                    }
                    if settings.blemish > 0.0 || settings.nasolabial > 0.0 {
                        let ring_radius = (geometry.width * 0.018).max(settings.radius * 1.5).clamp(1.0, 64.0);
                        let surround = source.ring(sample_point, ring_radius, color);
                        let red = redness(color) - redness(surround);
                        let automatic = smoothstep(0.018, 0.11, red) * (1.0 - smoothstep(0.20, 0.45, color_distance(color, surround)));
                        let manual = geometry.repair_weight(point);
                        result = mix(result, surround, settings.blemish * automatic.max(manual) * area * 0.92);
                        let shadow = (luma(surround) - luma(color)).max(0.0);
                        let crease = geometry.fold_weight(point) * smoothstep(0.005, 0.08, shadow);
                        // Lift only the local shadow; retain high-frequency texture and chroma.
                        let lift = shadow.min(0.16 * (1.0 + luma(color).max(0.0))) * settings.nasolabial * crease * area * 0.8;
                        result = result.map(|v| v + lift);
                    }
                }
                // Additive midtone lightening has no SDR clamp; HDR highlights keep their headroom.
                let light = settings.whitening * area * 0.16 / (1.0 + luma(result).max(0.0));
                result = result.map(|v| v + light);
                let rose = settings.rosiness * area * 0.045;
                result[0] += rose;
                result[1] -= rose * 0.15;
                result[2] += rose * 0.12;
            }
            // Preserve destination alpha exactly, including during color-only cheek warping.
            let index = x * 4;
            for channel in 0..3 {
                let value = result[channel] * alpha;
                if value.is_finite() {
                    row[index + channel] = value;
                }
            }
        }
    });
    Ok(())
}

struct Source<'a> {
    pixels: &'a [[f32; 4]],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
}

impl Source<'_> {
    fn sample(&self, p: [f32; 2]) -> [f32; 4] {
        // Source dimensions are nonzero and the input coordinate is finite and locally bounded.
        let fx = (p[0] - 0.5 - self.x0 as f32).clamp(0.0, self.w.saturating_sub(1) as f32);
        let fy = (p[1] - 0.5 - self.y0 as f32).clamp(0.0, self.h.saturating_sub(1) as f32);
        let (x, y) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x + 1).min(self.w - 1), (y + 1).min(self.h - 1));
        let (tx, ty) = (fx - x as f32, fy - y as f32);
        // Snapshot allocation and all four clamped coordinates were validated before processing.
        let [a, b, c, d] = [self.pixels[y * self.w + x], self.pixels[y * self.w + x1], self.pixels[y1 * self.w + x], self.pixels[y1 * self.w + x1]];
        let mut out = [0.0; 4];
        for k in 0..4 {
            out[k] = (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty;
        }
        out
    }

    fn bilateral(&self, p: [f32; 2], center: [f32; 3], radius: f32) -> [f32; 3] {
        const TAPS: [[f32; 2]; 16] = [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.71, 0.71],
            [-0.71, 0.71],
            [0.71, -0.71],
            [-0.71, -0.71],
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.0, -0.5],
            [0.35, 0.35],
            [-0.35, 0.35],
            [0.35, -0.35],
            [-0.35, -0.35],
        ];
        let mut sum = center;
        let mut total = 1.0;
        let sigma = 0.065 * (1.0 + luma(center).max(0.0));
        for offset in TAPS {
            let pixel = self.sample([p[0] + offset[0] * radius, p[1] + offset[1] * radius]);
            if let Some(color) = straight(pixel) {
                let distance = color_distance(center, color) / sigma;
                let weight = pixel[3] / (1.0 + distance * distance * distance * distance);
                for k in 0..3 {
                    sum[k] += color[k] * weight;
                }
                total += weight;
            }
        }
        sum.map(|v| v / total)
    }

    fn ring(&self, p: [f32; 2], radius: f32, fallback: [f32; 3]) -> [f32; 3] {
        const TAPS: [[f32; 2]; 8] = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]];
        let mut sum = [0.0; 3];
        let mut total = 0.0;
        for offset in TAPS {
            let pixel = self.sample([p[0] + offset[0] * radius, p[1] + offset[1] * radius]);
            if let Some(color) = straight(pixel) {
                for k in 0..3 {
                    sum[k] += color[k] * pixel[3];
                }
                total += pixel[3];
            }
        }
        if total > 1e-6 { sum.map(|v| v / total) } else { fallback }
    }
}

struct Geometry<'a> {
    face: &'a FaceRegion,
    size: [f32; 2],
    center: [f32; 2],
    width: f32,
    height: f32,
    points: [[f32; 2]; 10],
}

impl<'a> Geometry<'a> {
    fn new(face: &'a FaceRegion, w: f32, h: f32) -> Self {
        let [x, y, bw, bh] = face.bounds;
        Self {
            face,
            size: [w, h],
            center: [(x + bw * 0.5) * w, (y + bh * 0.5) * h],
            width: bw * w,
            height: bh * h,
            points: face.landmarks.map(|p| [p[0] * w, p[1] * h]),
        }
    }

    fn mask(&self, p: [f32; 2]) -> f32 {
        for [x, y, w, h] in &self.face.exclusions {
            if p[0] >= x * self.size[0] && p[0] < (x + w) * self.size[0] && p[1] >= y * self.size[1] && p[1] < (y + h) * self.size[1] {
                return 0.0;
            }
        }
        let q = [(p[0] - self.center[0]) / (self.width * 0.5), (p[1] - self.center[1]) / (self.height * 0.5)];
        1.0 - smoothstep(0.72, 1.0, q[0] * q[0] + q[1] * q[1])
    }

    fn feature_weight(&self, p: [f32; 2]) -> f32 {
        let left_eye = ellipse(p, self.points[EYE_LEFT], self.width * 0.13, self.height * 0.065);
        let right_eye = ellipse(p, self.points[EYE_RIGHT], self.width * 0.13, self.height * 0.065);
        let nose = ellipse(p, self.points[NOSE_TIP], self.width * 0.065, self.height * 0.045);
        let left = self.points[MOUTH_LEFT];
        let right = self.points[MOUTH_RIGHT];
        let middle = [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5];
        let mouth = ellipse(p, middle, (right[0] - left[0]).abs() * 0.6 + self.width * 0.02, self.height * 0.055);
        1.0 - left_eye.max(right_eye).max(nose).max(mouth)
    }

    fn warp(&self, p: [f32; 2], amount: f32, mask: f32) -> [f32; 2] {
        if amount == 0.0 {
            return p;
        }
        let mut delta = 0.0;
        for index in [CHEEK_LEFT, CHEEK_RIGHT] {
            let cheek = self.points[index];
            let influence = ellipse(p, cheek, self.width * 0.32, self.height * 0.30);
            let direction = if cheek[0] < self.center[0] { -1.0 } else { 1.0 };
            delta += direction * influence * self.width * 0.065 * amount;
        }
        [p[0] + delta * mask * self.feature_weight(p), p[1]]
    }

    fn repair_weight(&self, p: [f32; 2]) -> f32 {
        self.face.repair_spots.iter().fold(0.0_f32, |weight, spot| {
            weight.max(ellipse(p, [spot.center[0] * self.size[0], spot.center[1] * self.size[1]], spot.radius * self.width, spot.radius * self.width))
        })
    }

    fn fold_weight(&self, p: [f32; 2]) -> f32 {
        let mut result = 0.0_f32;
        for (nose, mouth) in [(NOSE_LEFT, MOUTH_LEFT), (NOSE_RIGHT, MOUTH_RIGHT)] {
            let a = self.points[nose];
            let b = self.points[mouth];
            let delta = [b[0] - a[0], b[1] - a[1]];
            let length_sq = delta[0] * delta[0] + delta[1] * delta[1];
            if length_sq <= 1e-6 {
                continue;
            }
            let t = (((p[0] - a[0]) * delta[0] + (p[1] - a[1]) * delta[1]) / length_sq).clamp(0.0, 1.0);
            let closest = [a[0] + delta[0] * t, a[1] + delta[1] * t];
            let distance = ((p[0] - closest[0]).powi(2) + (p[1] - closest[1]).powi(2)).sqrt() / (self.width * 0.04).max(0.5);
            result = result.max((1.0 - smoothstep(0.3, 1.0, distance)) * smoothstep(0.0, 0.18, t) * (1.0 - smoothstep(0.8, 1.0, t)));
        }
        result
    }
}

fn straight(p: [f32; 4]) -> Option<[f32; 3]> {
    if p[3] <= 1e-6 {
        return None;
    }
    let color = [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
    color.iter().all(|v| v.is_finite()).then_some(color)
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn redness(c: [f32; 3]) -> f32 {
    c[0] - 0.5 * (c[1] + c[2])
}

fn color_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn mix(a: [f32; 3], b: [f32; 3], weight: f32) -> [f32; 3] {
    let mut out = a;
    for k in 0..3 {
        out[k] = a[k] * (1.0 - weight) + b[k] * weight;
    }
    out
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn ellipse(p: [f32; 2], center: [f32; 2], rx: f32, ry: f32) -> f32 {
    let x = (p[0] - center[0]) / rx.max(1e-5);
    let y = (p[1] - center[1]) / ry.max(1e-5);
    1.0 - smoothstep(0.36, 1.0, x * x + y * y)
}

fn skin_weight(c: [f32; 3]) -> f32 {
    let positive = c.map(|v| v.max(0.0));
    let sum = positive[0] + positive[1] + positive[2];
    if !sum.is_finite() || sum <= 1e-6 {
        return 0.0;
    }
    let [r, g, b] = positive.map(|v| v / sum);
    // Chromaticity rather than brightness thresholds accommodates both light and dark complexions.
    smoothstep(-0.035, 0.025, r - b) * (1.0 - smoothstep(0.10, 0.22, g - r))
}
