//! Shared evaluated Beauty parameters for CPU rendering and the GPU frame plan.

use filmcraft_project::beauty as data;
use filmcraft_project::{EffectInstance, ParamValue};

use crate::beauty::{BeautySettings, FaceRegion, RepairSpot};
use crate::effects::FxCtx;

pub fn evaluate(e: &EffectInstance, cx: &FxCtx) -> Option<(BeautySettings, FaceRegion)> {
    if e.effect != "beauty" || !e.enabled {
        return None;
    }
    let amount = |id: &str| e.f64_at(id, cx.t) as f32 / 100.0;
    let settings = BeautySettings {
        smoothing: amount("smoothing"),
        whitening: amount("whitening"),
        rosiness: amount("rosiness"),
        blemish: amount("blemish"),
        nasolabial: amount("nasolabial"),
        slimming: amount("slimming"),
        detail: amount("detail"),
        radius: e.f64_at("radius", cx.t) as f32 * if cx.px_scale.is_finite() { cx.px_scale.max(0.0) } else { 1.0 },
        skin_only: e.param("skin_only").and_then(|p| p.value.as_bool()).unwrap_or(true),
    }
    .sanitized();
    if settings.is_identity() {
        return None;
    }
    let targeted = data::person(e).is_some();
    if targeted && cx.env.is_some_and(|env| !env.beauty_source_compatible()) {
        log::warn!("Beauty person {} was skipped because this clip has effects that change analysed source coordinates", data::person(e).unwrap_or_default());
        return None;
    }
    if targeted && data::confidence_at(e, cx.t) < data::MIN_CONFIDENCE {
        return None;
    }
    let mut face = match data::geometry_at(e, cx.t) {
        Ok(Some(g)) => {
            // A detected/tracked face may partly leave the source image. Restrict every operation
            // to its visible portion; the CPU and shader receive exactly these same bounds.
            let [x0, y0, x1, y1] = g.bounds.map(|v| v.clamp(0.0, 1.0) as f32);
            if x1 - x0 < 0.005 || y1 - y0 < 0.005 {
                return None;
            }
            let mut landmarks = [[0.0; 2]; 10];
            for (dst, p) in landmarks.iter_mut().zip(g.landmarks) {
                *dst = [(p[0] as f32).clamp(x0, x1), (p[1] as f32).clamp(y0, y1)];
            }
            FaceRegion { bounds: [x0, y0, x1 - x0, y1 - y0], landmarks, repair_spots: Vec::new(), exclusions: Vec::new() }
        }
        Ok(None) if !targeted => FaceRegion::from_bounds([0.0, 0.0, 1.0, 1.0]),
        Ok(None) => return None,
        Err(error) => {
            log::warn!("Beauty target geometry is invalid: {error}");
            return None;
        }
    };
    if targeted && let Some(env) = cx.env {
        let [x, y, w, h] = face.bounds;
        for peer in env.beauty_peers(e) {
            let [x0, y0, x1, y1] = peer.bounds.map(|v| v.clamp(0.0, 1.0) as f32);
            if x1 > x0 && y1 > y0 && x0 < x + w && x1 > x && y0 < y + h && y1 > y {
                face.exclusions.push([x0, y0, x1 - x0, y1 - y0]);
                break;
            }
        }
    }
    if let Some(p) = e.param("repair_spots") {
        match p.value_at(cx.t) {
            ParamValue::Path(path) if path.vertices.len() <= 64 => {
                for v in path.vertices {
                    face.repair_spots.push(RepairSpot {
                        center: [face.bounds[0] + v.p.x as f32 * face.bounds[2], face.bounds[1] + v.p.y as f32 * face.bounds[3]],
                        radius: v.t_in.x as f32,
                    });
                }
            }
            _ => {
                log::warn!("Beauty repair spots have invalid storage or exceed 64 points");
                return None;
            }
        }
    }
    if let Err(error) = face.validate() {
        log::warn!("Beauty face is invalid: {error}");
        return None;
    }
    Some((settings, face))
}

pub fn apply(image: &mut crate::Image, e: &EffectInstance, cx: &FxCtx) {
    if let Some((settings, face)) = evaluate(e, cx)
        && let Err(error) = crate::beauty::process(image, settings, Some(&face))
    {
        log::warn!("Beauty could not render person {}: {error}", data::person(e).unwrap_or("whole frame"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfx::{FxEnv, ItemEnv};
    use filmcraft_geom::Affine;
    use filmcraft_project::{ItemKind, Label, Project, SequenceSettings, TrackKind, find_effect};
    use filmcraft_time::{FrameRate, Tick, TimeRange};

    #[test]
    fn flipped_faces_never_receive_the_other_persons_source_coordinate_beauty() {
        let mut project = Project::new("Beauty coordinate safety");
        let sequence = project.new_sequence("sequence", SequenceSettings::default(), 1, 0, None);
        let source = project.add_item(
            "source",
            Label::Iris,
            ItemKind::AdjustmentLayer { width: 128, height: 64, rate: FrameRate::FPS_24, duration: Tick::from_seconds_f64(1.0) },
            None,
        );
        let mut item =
            project.make_track_item(source, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, Tick::from_seconds_f64(1.0)), FrameRate::FPS_24).unwrap();
        let flip = find_effect("horizontal_flip").unwrap().instance();
        let mut a = find_effect("beauty").unwrap().instance();
        let mut b = find_effect("beauty").unwrap().instance();
        data::assign(&mut a, "A", &data::FaceGeometry::from_bounds([0.05, 0.1, 0.42, 0.9]).unwrap(), 1.0).unwrap();
        data::assign(&mut b, "B", &data::FaceGeometry::from_bounds([0.58, 0.1, 0.95, 0.9]).unwrap(), 1.0).unwrap();
        a.param_mut("whitening").unwrap().value = ParamValue::Float(100.0);
        item.effects.extend([flip.clone(), a.clone(), b]);
        let sources = crate::SourceMap::default();
        let seq = project.sequence(sequence).unwrap();
        let env = ItemEnv {
            project: &project,
            seq,
            item: &item,
            t: Tick::ZERO,
            opts: crate::RenderOptions::default(),
            sources: &sources,
            want: 1.0,
            layer_size: (128, 64),
            layer_to_output: Affine::IDENTITY,
            tc: "",
        };
        assert!(!env.beauty_source_compatible());
        let cx = FxCtx {
            t: Tick::ZERO,
            px_scale: 1.0,
            seconds: 0.0,
            timecode: "",
            clip_name: "two faces",
            project: Some(&project),
            env: Some(&env),
            frame_rate: 24.0,
            working: filmcraft_color::WorkingSpace::Rec709,
        };
        let mut image = crate::Image::filled(128, 64, [0.4, 0.25, 0.18, 1.0]);
        for row in image.px.as_chunks_mut::<512>().0 {
            for pixel in row[256..].as_chunks_mut::<4>().0 {
                *pixel = [0.18, 0.09, 0.05, 1.0];
            }
        }
        crate::effects::apply(&mut image, &flip, &cx);
        let flipped = image.clone();
        assert_eq!(flipped.get(30, 32), [0.18, 0.09, 0.05, 1.0], "person B now occupies A's stored coordinates");
        assert!(evaluate(&a, &cx).is_none());
        assert!(crate::gpufx::FxOp::eval(&a, &cx, 128, 64).is_none(), "GPU planning must share the source-coordinate guard");
        apply(&mut image, &a, &cx);
        assert_eq!(image, flipped, "stale A geometry must never retouch B after a flip");

        let mut compatible = item.clone();
        compatible.effects.iter_mut().find(|effect| effect.effect == "horizontal_flip").unwrap().enabled = false;
        let compatible_env = ItemEnv { item: &compatible, ..env };
        let compatible_cx = FxCtx { env: Some(&compatible_env), ..cx };
        assert!(compatible_env.beauty_source_compatible(), "disabled geometric effects cannot move face coordinates");
        assert!(evaluate(&a, &compatible_cx).is_some());
        apply(&mut image, &a, &compatible_cx);
        assert_ne!(image, flipped, "compatible chains still apply Beauty");
    }
}
