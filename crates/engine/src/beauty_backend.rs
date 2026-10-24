//! The optional neural face-analysis adapter; no model downloads happen during playback.

use crate::beauty::FaceAnalyzer;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn model_dir(s: &crate::Session) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("FILMCRAFT_FACE_MODEL_DIR").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(path));
    }
    s.prefs_path
        .as_ref()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .or_else(crate::autosave::default_data_dir)
        .map(|p| p.join("models").join("mediapipe-face"))
}

pub fn available() -> bool {
    cfg!(all(feature = "face-analysis", not(target_arch = "wasm32")))
}

pub fn installed(dir: &Path) -> bool {
    filmcraft_face_analysis::models::FILES.iter().all(|f| std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.is_file() && m.len() == f.size))
}

#[cfg(all(feature = "face-analysis", not(target_arch = "wasm32")))]
struct Adapter(Box<dyn filmcraft_face_analysis::FaceAnalyzer>);

#[cfg(all(feature = "face-analysis", not(target_arch = "wasm32")))]
impl FaceAnalyzer for Adapter {
    fn analyze(&self, width: usize, height: usize, rgba: &[u8]) -> std::result::Result<Vec<crate::beauty::FaceObservation>, String> {
        use filmcraft_face_analysis::{AnalysisOptions, RgbImage};
        let width = u32::try_from(width).map_err(|_| "face analysis width overflows")?;
        let height = u32::try_from(height).map_err(|_| "face analysis height overflows")?;
        let faces = self
            .0
            .analyze(RgbImage { width, height, channels: 4, pixels: rgba }, AnalysisOptions { max_faces: 8, min_confidence: 0.5 })
            .map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(faces.len().min(8));
        for face in faces.into_iter().take(8) {
            let point = |index: usize| -> std::result::Result<[f64; 2], String> {
                let p = face.landmarks.get(index).ok_or_else(|| format!("face model returned no landmark {index}"))?;
                Ok([p.x as f64, p.y as f64])
            };
            let mean = |a: usize, b: usize| -> std::result::Result<[f64; 2], String> {
                let a = point(a)?;
                let b = point(b)?;
                Ok([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5])
            };
            let landmarks =
                [mean(33, 133)?, mean(263, 362)?, point(1)?, point(61)?, point(291)?, point(152)?, point(234)?, point(454)?, point(98)?, point(327)?];
            // Mesh landmarks can extend beyond the detector's crop, especially at the chin.
            let bounds = [
                landmarks.iter().map(|p| p[0]).fold(face.bounds.left as f64, f64::min),
                landmarks.iter().map(|p| p[1]).fold(face.bounds.top as f64, f64::min),
                landmarks.iter().map(|p| p[0]).fold(face.bounds.right as f64, f64::max),
                landmarks.iter().map(|p| p[1]).fold(face.bounds.bottom as f64, f64::max),
            ];
            let geometry = filmcraft_project::beauty::FaceGeometry { bounds, landmarks };
            geometry.validate()?;
            out.push(crate::beauty::FaceObservation { geometry, confidence: face.confidence as f64 });
        }
        Ok(out)
    }
}

pub fn analyzer(dir: Option<&Path>) -> std::result::Result<Arc<dyn FaceAnalyzer>, String> {
    #[cfg(all(feature = "face-analysis", not(target_arch = "wasm32")))]
    {
        use std::sync::{Mutex, OnceLock};
        type Cache = Option<(PathBuf, Arc<dyn FaceAnalyzer>)>;
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        let dir = dir.ok_or("face model directory is unavailable")?;
        let cache = CACHE.get_or_init(Mutex::default);
        let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((path, analyzer)) = cache.as_ref()
            && path == dir
        {
            return Ok(analyzer.clone());
        }
        if !installed(dir) {
            return Err("Face models are not installed. Use Download Face Models, then Detect Faces; manual A/B regions are also available.".into());
        }
        let analyzer: Arc<dyn FaceAnalyzer> =
            Arc::new(Adapter(Box::new(filmcraft_face_analysis::mediapipe::MediaPipeAnalyzer::from_dir(dir).map_err(|e| e.to_string())?)));
        *cache = Some((dir.to_path_buf(), analyzer.clone()));
        Ok(analyzer)
    }
    #[cfg(not(all(feature = "face-analysis", not(target_arch = "wasm32"))))]
    {
        let _ = dir;
        Err("Automatic face detection requires a desktop build with the face-analysis feature. Manual A/B face regions and tracking are available.".into())
    }
}
