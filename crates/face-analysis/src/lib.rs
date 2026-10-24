//! Offline portrait analysis, independent of the editor and rendering layers.
#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

#[cfg(feature = "mediapipe")]
mod fold;
#[cfg(feature = "mediapipe")]
pub mod mediapipe;
pub mod models;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FaceError {
    #[error("invalid face analysis input: {0}")]
    InvalidInput(String),
    #[error("face analysis model is not installed: {0}")]
    NotInstalled(String),
    #[error("face analysis is unavailable: {0}")]
    Unavailable(String),
    #[error("face analysis model: {0}")]
    Model(String),
    #[error("face analysis I/O: {0}")]
    Io(String),
}

/// Tightly packed RGB8 or RGBA8, top to bottom; alpha does not affect detection.
#[derive(Debug, Clone, Copy)]
pub struct RgbImage<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub pixels: &'a [u8],
}

impl RgbImage<'_> {
    pub fn validate(&self) -> Result<(), FaceError> {
        let count = u64::from(self.width).checked_mul(u64::from(self.height));
        let Some(count) = count.filter(|n| *n > 0 && *n <= 16_777_216) else {
            return Err(FaceError::InvalidInput("image must contain 1..=16777216 pixels".into()));
        };
        if !matches!(self.channels, 3 | 4) {
            return Err(FaceError::InvalidInput("expected RGB8 or RGBA8".into()));
        }
        let expected = count.checked_mul(u64::from(self.channels)).and_then(|n| usize::try_from(n).ok());
        if expected != Some(self.pixels.len()) {
            return Err(FaceError::InvalidInput("image byte length does not match dimensions".into()));
        }
        Ok(())
    }
}

/// Normalized, axis-aligned source image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBounds {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

/// Source-normalized MediaPipe Face Mesh coordinates (468 landmark indices).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceDetection {
    pub bounds: FaceBounds,
    pub landmarks: Vec<Landmark>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AnalysisOptions {
    pub max_faces: usize,
    pub min_confidence: f32,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self { max_faces: 2, min_confidence: 0.5 }
    }
}

impl AnalysisOptions {
    pub fn validate(&self) -> Result<(), FaceError> {
        if !(1..=8).contains(&self.max_faces) || !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(FaceError::InvalidInput("max_faces must be 1..=8 and confidence finite in 0..=1".into()));
        }
        Ok(())
    }
}

pub trait FaceAnalyzer: Send + Sync {
    fn analyze(&self, image: RgbImage<'_>, options: AnalysisOptions) -> Result<Vec<FaceDetection>, FaceError>;
}

/// Construct the offline backend from `FILMCRAFT_FACE_MODEL_DIR`.
pub fn default_analyzer() -> Result<Box<dyn FaceAnalyzer>, FaceError> {
    #[cfg(feature = "mediapipe")]
    {
        let path = std::env::var_os("FILMCRAFT_FACE_MODEL_DIR")
            .ok_or_else(|| FaceError::NotInstalled("set FILMCRAFT_FACE_MODEL_DIR to the verified MediaPipe model directory".into()))?;
        Ok(Box::new(mediapipe::MediaPipeAnalyzer::from_dir(std::path::Path::new(&path))?))
    }
    #[cfg(not(feature = "mediapipe"))]
    Err(FaceError::Unavailable("rebuild with the face-analysis mediapipe feature".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostile_image_dimensions_and_lengths_return_errors() {
        for (width, height, channels, pixels) in
            [(0, 1, 3, &[][..]), (1, 0, 3, &[][..]), (u32::MAX, u32::MAX, 4, &[][..]), (1, 1, 2, &[0; 2][..]), (1, 1, 4, &[0; 3][..]), (1, 1, 3, &[0; 4][..])]
        {
            let result = std::panic::catch_unwind(|| RgbImage { width, height, channels, pixels }.validate());
            assert!(result.expect("image validator must not unwind").is_err());
        }
        assert!(RgbImage { width: 1, height: 1, channels: 3, pixels: &[1, 2, 3] }.validate().is_ok());
        assert!(RgbImage { width: 1, height: 1, channels: 4, pixels: &[1, 2, 3, 0] }.validate().is_ok());
    }

    #[test]
    fn hostile_options_return_errors() {
        for max_faces in [0, 9, usize::MAX] {
            assert!(AnalysisOptions { max_faces, ..Default::default() }.validate().is_err());
        }
        for min_confidence in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert!(AnalysisOptions { min_confidence, ..Default::default() }.validate().is_err());
        }
    }
}
