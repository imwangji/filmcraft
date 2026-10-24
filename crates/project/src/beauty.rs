//! Per-person beauty targets. Geometry is stored as ordinary, keyframed effect parameters so
//! previews, exports, undo and project files all read the same analysis. Coordinates are normalised
//! source-image coordinates, independent of playback resolution and Motion.

use filmcraft_geom::Vec2;
use filmcraft_time::Tick;

use crate::{EffectInstance, MaskPath, MaskVertex, Param, ParamValue};

pub const PERSON: &str = "person";
pub const FACE_PATH: &str = "face_path";
pub const CONFIDENCE: &str = "face_confidence";
pub const MAX_PERSONS: usize = 2;
pub const LANDMARK_COUNT: usize = 10;
pub const MIN_CONFIDENCE: f64 = 0.5;

/// Source face coordinates are valid only while ordinary effects retain picture geometry.
/// Intrinsic Motion is applied by the renderer afterwards and is mapped by the monitor overlay.
pub fn source_coordinate_conflict(effects: &[EffectInstance]) -> Option<&EffectInstance> {
    effects.iter().find(|effect| {
        if !effect.enabled || effect.effect == "beauty" {
            return false;
        }
        let Some(def) = effect.def() else { return true };
        if def.intrinsic || def.kind != crate::EffectKind::Video {
            return false;
        }
        if def.id == "crop" && effect.param("zoom").and_then(|p| p.value.as_bool()).unwrap_or(false) {
            return true;
        }
        let category = def.category.get(1).copied().unwrap_or("");
        let preserves_geometry =
            matches!(category, "Adjust" | "Blur & Sharpen" | "Color Correction" | "Image Control" | "Keying" | "Noise & Grain" | "Lights & Glows")
                || matches!(
                    def.id,
                    "crop"
                        | "edge_feather"
                        | "brightness_contrast"
                        | "proc_amp"
                        | "tint"
                        | "gamma"
                        | "levels"
                        | "color_balance"
                        | "black_white"
                        | "invert"
                        | "posterize"
                        | "alpha_adjust"
                        | "gaussian_blur"
                        | "lumetri"
                );
        !preserves_geometry
    })
}

/// [left, top, right, bottom], followed by the semantic points used by the local retoucher.
/// Landmarks: left/right eye, nose tip, left/right mouth, chin, left/right cheek, left/right nostril.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometry {
    pub bounds: [f64; 4],
    pub landmarks: [[f64; 2]; LANDMARK_COUNT],
}

impl FaceGeometry {
    /// A manually selected region. These are geometric estimates, not detected facial landmarks.
    pub fn from_bounds(bounds: [f64; 4]) -> Result<Self, String> {
        let [x0, y0, x1, y1] = bounds;
        if !bounds.iter().all(|v| v.is_finite() && (-0.5..=1.5).contains(v)) || x1 - x0 < 0.005 || y1 - y0 < 0.005 {
            return Err("face bounds must be finite, ordered normalised coordinates with positive size".into());
        }
        let p = |x: f64, y: f64| [x0 + x * (x1 - x0), y0 + y * (y1 - y0)];
        let geometry = Self {
            bounds,
            landmarks: [
                p(0.32, 0.36),
                p(0.68, 0.36),
                p(0.5, 0.55),
                p(0.35, 0.73),
                p(0.65, 0.73),
                p(0.5, 0.96),
                p(0.08, 0.58),
                p(0.92, 0.58),
                p(0.42, 0.58),
                p(0.58, 0.58),
            ],
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn validate(&self) -> Result<(), String> {
        let [x0, y0, x1, y1] = self.bounds;
        if !self.bounds.iter().all(|v| v.is_finite() && (-0.5..=1.5).contains(v)) || x1 - x0 < 0.005 || y1 - y0 < 0.005 {
            return Err("invalid beauty face bounds".into());
        }
        if !self.landmarks.iter().flatten().all(|v| v.is_finite() && (-0.5..=1.5).contains(v)) {
            return Err("beauty landmarks must be finite normalised coordinates".into());
        }
        Ok(())
    }

    pub fn to_path(&self) -> MaskPath {
        let [x0, y0, x1, y1] = self.bounds;
        let mut vertices = Vec::with_capacity(LANDMARK_COUNT + 2);
        vertices.push(MaskVertex::corner(Vec2::new(x0, y0)));
        vertices.push(MaskVertex::corner(Vec2::new(x1, y1)));
        vertices.extend(self.landmarks.iter().map(|p| MaskVertex::corner(Vec2::new(p[0], p[1]))));
        MaskPath { vertices, closed: false }
    }

    pub fn from_path(path: &MaskPath) -> Result<Self, String> {
        if path.vertices.len() != LANDMARK_COUNT + 2 {
            return Err("beauty face geometry needs bounds and ten semantic landmarks".into());
        }
        let a = path.vertices.first().ok_or("missing face bounds")?.p;
        let b = path.vertices.get(1).ok_or("missing face bounds")?.p;
        let mut landmarks = [[0.0; 2]; LANDMARK_COUNT];
        for (dst, src) in landmarks.iter_mut().zip(path.vertices.iter().skip(2)) {
            *dst = [src.p.x, src.p.y];
        }
        let geometry = Self { bounds: [a.x, a.y, b.x, b.y], landmarks };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn contains(&self, p: [f64; 2]) -> bool {
        let [x0, y0, x1, y1] = self.bounds;
        p[0].is_finite() && p[1].is_finite() && (x0..=x1).contains(&p[0]) && (y0..=y1).contains(&p[1])
    }

    pub fn centre(&self) -> [f64; 2] {
        [(self.bounds[0] + self.bounds[2]) * 0.5, (self.bounds[1] + self.bounds[3]) * 0.5]
    }
}

/// Stable A/B identity is separate from the face's current left-to-right order.
pub fn person(effect: &EffectInstance) -> Option<&str> {
    if effect.effect != "beauty" {
        return None;
    }
    match effect.param(PERSON).map(|p| &p.value) {
        Some(ParamValue::Text(name)) if name == "A" || name == "B" => Some(name),
        _ => None,
    }
}

pub fn geometry_at(effect: &EffectInstance, time: Tick) -> Result<Option<FaceGeometry>, String> {
    let Some(param) = effect.param(FACE_PATH) else { return Ok(None) };
    match param.value_at(time) {
        ParamValue::Path(path) => FaceGeometry::from_path(&path).map(Some),
        _ => Err("beauty face geometry has the wrong parameter type".into()),
    }
}

pub fn confidence_at(effect: &EffectInstance, time: Tick) -> f64 {
    // Never reuse the first/last analysed face outside the analysed interval: that can apply A's
    // settings to B after A leaves the picture. A manually assigned static region has no keys.
    if let Some(param) = effect.param(CONFIDENCE)
        && !param.keyframes.is_empty()
        && (param.keyframes.first().is_some_and(|k| time < k.time) || param.keyframes.last().is_some_and(|k| time > k.time))
    {
        return 0.0;
    }
    let v = effect.f64_at(CONFIDENCE, time);
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

pub fn assign(effect: &mut EffectInstance, name: &str, geometry: &FaceGeometry, confidence: f64) -> Result<(), String> {
    if effect.effect != "beauty" || !matches!(name, "A" | "B") || !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err("beauty target must be A or B with a confidence in 0..1".into());
    }
    geometry.validate()?;
    effect.params.insert(PERSON.into(), Param::new(ParamValue::Text(name.into())));
    effect.params.insert(FACE_PATH.into(), Param::new(ParamValue::Path(geometry.to_path())));
    effect.params.insert(CONFIDENCE.into(), Param::new(ParamValue::Float(confidence)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinate_conflicts_cover_flips_and_zooming_crops_but_allow_motion_and_grading() {
        let mut effects = crate::effect::intrinsic_video();
        effects.push(crate::find_effect("lumetri").unwrap().instance());
        effects.push(crate::find_effect("gaussian_blur").unwrap().instance());
        assert!(source_coordinate_conflict(&effects).is_none());
        effects.push(crate::find_effect("crop").unwrap().instance());
        assert!(source_coordinate_conflict(&effects).is_none());
        effects.last_mut().unwrap().param_mut("zoom").unwrap().value = ParamValue::Bool(true);
        assert_eq!(source_coordinate_conflict(&effects).unwrap().effect, "crop");
        effects.last_mut().unwrap().enabled = false;
        assert!(source_coordinate_conflict(&effects).is_none());
        effects.push(crate::find_effect("horizontal_flip").unwrap().instance());
        assert_eq!(source_coordinate_conflict(&effects).unwrap().effect, "horizontal_flip");
    }

    #[test]
    fn geometry_round_trip_and_hostile_values() {
        let g = FaceGeometry::from_bounds([0.1, 0.2, 0.4, 0.8]).unwrap();
        assert_eq!(FaceGeometry::from_path(&g.to_path()).unwrap(), g);
        assert!(FaceGeometry::from_bounds([0.5, 0.5, 0.2, 0.2]).is_err());
        assert!(FaceGeometry::from_bounds([f64::NAN, 0.0, 1.0, 1.0]).is_err());
        assert!(FaceGeometry::from_bounds([0.0, 0.0, f64::INFINITY, 1.0]).is_err());
        assert!(FaceGeometry::from_path(&MaskPath::default()).is_err());
    }

    #[test]
    fn two_targets_and_geometry_survive_serialisation() {
        let def = crate::find_effect("beauty").unwrap();
        let mut a = def.instance();
        let mut b = def.instance();
        assign(&mut a, "A", &FaceGeometry::from_bounds([0.05, 0.1, 0.4, 0.9]).unwrap(), 0.9).unwrap();
        assign(&mut b, "B", &FaceGeometry::from_bounds([0.6, 0.1, 0.95, 0.9]).unwrap(), 0.8).unwrap();
        a.param_mut("whitening").unwrap().value = ParamValue::Float(80.0);
        assert_eq!(b.f64_at("whitening", Tick(0)), 0.0);
        let saved = serde_json::to_string(&vec![a.clone(), b.clone()]).unwrap();
        let loaded: Vec<EffectInstance> = serde_json::from_str(&saved).unwrap();
        assert_eq!(person(&loaded[0]), Some("A"));
        assert_eq!(person(&loaded[1]), Some("B"));
        assert_eq!(loaded[0].f64_at("whitening", Tick(0)), 80.0);
        assert_eq!(loaded[1].f64_at("whitening", Tick(0)), 0.0);
        for (saved, original) in loaded.iter().zip([a, b]) {
            let saved = geometry_at(saved, Tick(0)).unwrap().unwrap();
            let original = geometry_at(&original, Tick(0)).unwrap().unwrap();
            for (a, b) in saved.bounds.iter().chain(saved.landmarks.iter().flatten()).zip(original.bounds.iter().chain(original.landmarks.iter().flatten())) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn tracked_face_is_hidden_outside_analysis_and_during_a_lost_interval() {
        let mut e = crate::find_effect("beauty").unwrap().instance();
        assign(&mut e, "A", &FaceGeometry::from_bounds([0.1, 0.1, 0.4, 0.9]).unwrap(), 1.0).unwrap();
        let p = e.param_mut(CONFIDENCE).unwrap();
        p.put_keyframe(Tick(10), ParamValue::Float(1.0));
        p.put_keyframe(Tick(20), ParamValue::Float(0.0));
        p.put_keyframe(Tick(30), ParamValue::Float(1.0));
        for k in &mut p.keyframes {
            k.interp = crate::Interpolation::Hold;
        }
        assert_eq!(confidence_at(&e, Tick(0)), 0.0);
        assert_eq!(confidence_at(&e, Tick(15)), 1.0);
        assert_eq!(confidence_at(&e, Tick(25)), 0.0);
        assert_eq!(confidence_at(&e, Tick(40)), 0.0);
    }
}
