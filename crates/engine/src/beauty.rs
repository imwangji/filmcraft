//! Independent A/B beauty targets, source-frame analysis and undoable tracking.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use filmcraft_geom::Vec2;
use filmcraft_project::beauty::{self as data, FaceGeometry};
use filmcraft_project::{ClipId, Interpolation, ItemId, Param, ParamValue, TrackItem, TrackKind};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, clip_p, has_seq, str_p, time_p};
use crate::{EngineError, Result, Session};

const MAX_FRAMES: usize = 18_000;
const MAX_ANALYSIS_PIXELS: usize = 4_194_304;
const MAX_SOURCE_PIXELS: usize = 67_108_864;

fn analysis_rgba(frame: &filmcraft_frame::VideoFrame) -> std::result::Result<(usize, usize, Vec<u8>), String> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let pixels = w.checked_mul(h).filter(|n| *n > 0 && *n <= MAX_SOURCE_PIXELS).ok_or("face analysis source exceeds its size limit")?;
    let expected = pixels.checked_mul(4).ok_or("face analysis source size overflows")?;
    let rgba = frame.to_rgba8();
    if rgba.len() != expected {
        return Err("face analysis frame has an invalid pixel buffer".into());
    }
    // FrameRequest.scale is a hint: still images and some video decoders return full size.
    // Always bound the actual inference/optical-flow input, including 4K and 8K sources.
    if w.max(h) <= 640 {
        return Ok((w, h, rgba));
    }
    let scale = 640.0 / w.max(h) as f64;
    let width = (w as f64 * scale).round().max(1.0) as u32;
    let height = (h as f64 * scale).round().max(1.0) as u32;
    let count = (width as usize).checked_mul(height as usize).ok_or("face analysis size overflows")?;
    if count > MAX_ANALYSIS_PIXELS {
        return Err("face analysis frame exceeds its size limit".into());
    }
    let source = image::RgbaImage::from_raw(frame.width, frame.height, rgba).ok_or("invalid face analysis source buffer")?;
    let small = image::imageops::resize(&source, width, height, image::imageops::FilterType::Triangle);
    Ok((width as usize, height as usize, small.into_raw()))
}

#[derive(Clone, Debug)]
pub struct FaceObservation {
    pub geometry: FaceGeometry,
    pub confidence: f64,
}

/// Injectable analysis boundary: tests can supply deterministic observations; production supplies
/// the optional, local MediaPipe backend. No network request is made by this interface.
pub trait FaceAnalyzer: Send + Sync {
    fn analyze(&self, width: usize, height: usize, rgba: &[u8]) -> std::result::Result<Vec<FaceObservation>, String>;
}

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: has_seq, run, journal }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        spec("beauty.add", "Add Person Beauty", r#"{"clip":id?,"person":"A|B","bounds":[x,y,width,height]}"#, add, true),
        spec("beauty.select", "Select Beauty Person", r#"{"clip":id?,"person":"A|B"}"#, select, false),
        spec("beauty.list", "List Beauty Persons", r#"{"clip":id?,"time":ticks?}"#, list, false),
        spec(
            "beauty.setGeometry",
            "Adjust Beauty Face",
            r#"{"clip":id?,"person":"A|B","bounds":[x,y,width,height]?,"landmarks":[[x,y]…]?,"time":ticks?,"merge":string?,"begin":bool?}"#,
            set_geometry,
            true,
        ),
        spec("beauty.detect", "Detect Beauty Faces", r#"{"clip":id?,"wait":bool?}"#, detect, true),
        spec("beauty.addRepairSpot", "Add Beauty Repair Spot", r#"{"clip":id?,"person":"A|B","point":[x,y],"radius":0..0.2?}"#, repair_spot, true),
        spec("beauty.clearRepairSpots", "Clear Beauty Repair Spots", r#"{"clip":id?,"person":"A|B"}"#, clear_spots, true),
        spec("beauty.modelStatus", "Beauty Face Model Status", r#"{}"#, model_status, false),
        spec("beauty.downloadModels", "Download Face Models", r#"{"wait":bool?}"#, download_models, false),
        spec(
            "beauty.track",
            "Track Beauty Faces",
            r#"{"clip":id?,"person":"A|B"?,"direction":"forward|backward"?,"frames":1..18000?,"landmarks":bool?,"wait":bool?}"#,
            track,
            true,
        ),
    ]
}

fn target_clip(s: &Session, p: &Value) -> Result<ClipId> {
    let clip = clip_p(p, "clip").or_else(|| s.state.selection.first().copied()).ok_or_else(|| bad("beauty", "select a video clip first"))?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (track, _) = q.find_item(clip).ok_or_else(|| bad("beauty", "no such clip"))?;
    if q.track(track).is_none_or(|t| t.kind != TrackKind::Video) {
        return Err(bad("beauty", "beauty needs a video clip"));
    }
    Ok(clip)
}

fn person_param(p: &Value) -> Result<&str> {
    match str_p(p, "person") {
        Some(name @ ("A" | "B")) => Ok(name),
        _ => Err(bad("beauty", "person must be A or B")),
    }
}

fn rect_param(p: &Value) -> Result<Option<[f64; 4]>> {
    let Some(v) = p.get("bounds") else { return Ok(None) };
    let values = v.as_array().filter(|a| a.len() == 4).ok_or_else(|| bad("beauty", "bounds must be [x,y,width,height] in normalised source coordinates"))?;
    let mut b = [0.0; 4];
    for (dst, src) in b.iter_mut().zip(values) {
        *dst = src.as_f64().filter(|v| v.is_finite()).ok_or_else(|| bad("beauty", "face bounds must be finite numbers"))?;
    }
    let out = [b[0], b[1], b[0] + b[2], b[1] + b[3]];
    FaceGeometry::from_bounds(out).map_err(|e| bad("beauty", e))?;
    Ok(Some(out))
}

fn add(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let name = person_param(p)?.to_string();
    let bounds = rect_param(p)?.ok_or_else(|| bad("beauty.add", "select a face region with `bounds`"))?;
    let geometry = FaceGeometry::from_bounds(bounds).map_err(|e| bad("beauty.add", e))?;
    let def = filmcraft_project::find_effect("beauty").ok_or_else(|| bad("beauty.add", "beauty effect is unavailable"))?;
    s.edit_sequence("Add Person Beauty", |q, _, st| {
        let (_, it) = q.find_item_mut(clip).ok_or_else(|| bad("beauty.add", "clip is gone"))?;
        require_source_geometry(it)?;
        if it.effects.iter().any(|e| data::person(e) == Some(name.as_str())) {
            return Err(bad("beauty.add", format!("person {name} already exists; adjust its face region instead")));
        }
        let mut effect = def.instance();
        data::assign(&mut effect, &name, &geometry, 1.0).map_err(|e| bad("beauty.add", e))?;
        let index = it.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
        it.effects.insert(index, effect);
        st.selected_effect = crate::EffectSel::of(clip, &it.effects, index);
        st.selected_mask = None;
        Ok(())
    })?;
    list(s, &json!({"clip": clip.0}))
}

fn require_source_geometry(item: &TrackItem) -> Result<()> {
    if let Some(effect) = data::source_coordinate_conflict(&item.effects) {
        return Err(bad("beauty", format!("Disable geometric effects before per-person Beauty: {}. Intrinsic Motion is supported.", effect.effect)));
    }
    Ok(())
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let name = person_param(p)?;
    let (_, it) = s.active_sequence().and_then(|q| q.find_item(clip)).ok_or_else(|| bad("beauty.select", "clip is gone"))?;
    let index = it.effects.iter().position(|e| data::person(e) == Some(name)).ok_or_else(|| bad("beauty.select", "person has not been assigned a face"))?;
    s.state.selected_effect = crate::EffectSel::of(clip, &it.effects, index);
    s.state.selected_mask = None;
    Ok(json!({"person":name,"effect":index}))
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let (_, it) = s.active_sequence().and_then(|q| q.find_item(clip)).ok_or_else(|| bad("beauty.list", "clip is gone"))?;
    let time = it.effect_time_at(time_p(s, p, "").unwrap_or(s.playhead()));
    let mut persons = Vec::new();
    for (index, effect) in it.effects.iter().enumerate() {
        let Some(name) = data::person(effect) else { continue };
        let g = data::geometry_at(effect, time).map_err(|e| bad("beauty.list", e))?;
        let geometry = g.map(|g| {
            let [x0, y0, x1, y1] = g.bounds;
            json!({"bounds":[x0,y0,x1-x0,y1-y0],"landmarks":g.landmarks})
        });
        persons.push(json!({"person":name,"effect":index,"enabled":effect.enabled,"confidence":data::confidence_at(effect,time),"geometry":geometry}));
    }
    Ok(json!({"clip":clip.0,"persons":persons,"tracking":s.beauty_jobs.iter().any(|j| j.clip==clip)}))
}

fn set_geometry(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let name = person_param(p)?.to_string();
    let at = time_p(s, p, "").unwrap_or(s.playhead());
    let bounds = rect_param(p)?;
    let points = match p.get("landmarks") {
        None => None,
        Some(v) => {
            let a = v.as_array().filter(|a| a.len() == data::LANDMARK_COUNT).ok_or_else(|| bad("beauty.setGeometry", "need ten semantic landmarks"))?;
            let mut points = [[0.0; 2]; data::LANDMARK_COUNT];
            for (dst, src) in points.iter_mut().zip(a) {
                let pair = src.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad("beauty.setGeometry", "each landmark must be [x,y]"))?;
                for (v, x) in dst.iter_mut().zip(pair) {
                    *v = x.as_f64().filter(|v| v.is_finite()).ok_or_else(|| bad("beauty.setGeometry", "landmarks must be finite"))?;
                }
            }
            Some(points)
        }
    };
    if bounds.is_none() && points.is_none() {
        return Err(bad("beauty.setGeometry", "provide bounds or landmarks"));
    }
    let merge = str_p(p, "merge").filter(|m| m.len() <= 128).map(|m| format!("beauty:{}:{name}:{m}", clip.0));
    if bool_p(p, "begin").unwrap_or(false) {
        s.history.merge_key = None;
    }
    s.edit_sequence_as("Adjust Beauty Face", merge.as_deref(), |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or_else(|| bad("beauty.setGeometry", "clip is gone"))?;
        require_source_geometry(it)?;
        let time = it.effect_time_at(at);
        let effect =
            it.effects.iter_mut().find(|e| data::person(e) == Some(name.as_str())).ok_or_else(|| bad("beauty.setGeometry", "person is not assigned"))?;
        let old =
            data::geometry_at(effect, time).map_err(|e| bad("beauty.setGeometry", e))?.ok_or_else(|| bad("beauty.setGeometry", "face geometry is missing"))?;
        let mut geometry = old.clone();
        if let Some(b) = bounds {
            geometry.bounds = b;
            let [x0, y0, x1, y1] = old.bounds;
            for (new, old) in geometry.landmarks.iter_mut().zip(&old.landmarks) {
                *new = [b[0] + (old[0] - x0) / (x1 - x0) * (b[2] - b[0]), b[1] + (old[1] - y0) / (y1 - y0) * (b[3] - b[1])];
            }
        }
        if let Some(points) = points {
            geometry.landmarks = points;
        }
        geometry.validate().map_err(|e| bad("beauty.setGeometry", e))?;
        let path = effect.param_mut(data::FACE_PATH).ok_or_else(|| bad("beauty.setGeometry", "face geometry is missing"))?;
        path.set_at(time, ParamValue::Path(geometry.to_path()));
        if let Some(confidence) = effect.param_mut(data::CONFIDENCE) {
            // Reassignment at a distant frame must not make the unanalysed gap active.
            let first = confidence.keyframes.first().map(|k| k.time);
            let last = confidence.keyframes.last().map(|k| k.time);
            let guard = match (first, last) {
                (_, Some(last)) if time > last => last.0.checked_add(1).map(Tick).filter(|guard| *guard < time),
                (Some(first), _) if time < first => time.0.checked_add(1).map(Tick).filter(|guard| *guard < first),
                _ => None,
            };
            if let Some(guard) = guard {
                confidence.put_keyframe(guard, ParamValue::Float(0.0));
            }
            confidence.set_at(time, ParamValue::Float(1.0));
            for key in &mut confidence.keyframes {
                key.interp = Interpolation::Hold;
            }
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

#[derive(Clone)]
struct FaceKey {
    name: String,
    time: Tick,
    geometry: FaceGeometry,
    confidence: f64,
}

pub struct PendingBeauty {
    pub job: u64,
    pub clip: ClipId,
    sequence: ItemId,
    original: TrackItem,
    original_item: Option<filmcraft_project::ProjectItem>,
    original_media: Option<(ItemId, filmcraft_project::MediaClip, Option<filmcraft_time::TimeRange>)>,
    detect: bool,
    keys: Arc<Mutex<Option<Vec<FaceKey>>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn validate_observations(observations: &[FaceObservation]) -> std::result::Result<(), String> {
    if observations.len() > 8 {
        return Err("face analyzer returned more than eight faces".into());
    }
    for o in observations {
        o.geometry.validate()?;
        if !o.confidence.is_finite() || !(0.0..=1.0).contains(&o.confidence) {
            return Err("face analyzer returned invalid confidence".into());
        }
    }
    Ok(())
}

fn repair_spot(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let name = person_param(p)?.to_string();
    let pair = p
        .get("point")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 2)
        .ok_or_else(|| bad("beauty.addRepairSpot", "point must be [x,y] in normalised source coordinates"))?;
    let x = pair.first().and_then(Value::as_f64).filter(|v| v.is_finite()).ok_or_else(|| bad("beauty.addRepairSpot", "point must be finite"))?;
    let y = pair.get(1).and_then(Value::as_f64).filter(|v| v.is_finite()).ok_or_else(|| bad("beauty.addRepairSpot", "point must be finite"))?;
    let radius = match p.get("radius") {
        Some(v) => v.as_f64().ok_or_else(|| bad("beauty.addRepairSpot", "radius must be a number"))?,
        None => 0.025,
    };
    if !radius.is_finite() || radius <= 0.0 || radius > 0.2 {
        return Err(bad("beauty.addRepairSpot", "radius must be in (0,0.2], relative to face width"));
    }
    let ph = s.playhead();
    s.edit_sequence("Add Beauty Repair Spot", |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or_else(|| bad("beauty.addRepairSpot", "clip is gone"))?;
        require_source_geometry(it)?;
        let mt = it.effect_time_at(ph);
        let e = it.effects.iter_mut().find(|e| data::person(e) == Some(name.as_str())).ok_or_else(|| bad("beauty.addRepairSpot", "person is not assigned"))?;
        let g = data::geometry_at(e, mt).map_err(|e| bad("beauty.addRepairSpot", e))?.ok_or_else(|| bad("beauty.addRepairSpot", "face is not assigned"))?;
        if !g.contains([x, y]) {
            return Err(bad("beauty.addRepairSpot", "repair spot must be inside the selected face"));
        }
        let mut path = match e.param("repair_spots").map(|p| &p.value) {
            Some(ParamValue::Path(p)) => p.clone(),
            None => filmcraft_project::MaskPath::default(),
            _ => return Err(bad("beauty.addRepairSpot", "invalid repair spot storage")),
        };
        if path.vertices.len() >= 64 {
            return Err(bad("beauty.addRepairSpot", "a face can have at most 64 repair spots"));
        }
        // Face-relative positions follow translation/scale without tying a repair to a screen pixel.
        let [x0, y0, x1, y1] = g.bounds;
        let mut point = filmcraft_project::MaskVertex::corner(Vec2::new((x - x0) / (x1 - x0), (y - y0) / (y1 - y0)));
        point.t_in.x = radius;
        path.vertices.push(point);
        e.params.insert("repair_spots".into(), Param::new(ParamValue::Path(path)));
        // A spot selected by hand is an explicit request to repair; enable its amount if neutral.
        if e.f64_at("blemish", mt) <= 0.0 {
            e.params.entry("blemish".into()).or_insert_with(|| Param::new(ParamValue::Float(0.0))).set_at(mt, ParamValue::Float(50.0));
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn clear_spots(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let name = person_param(p)?.to_string();
    s.edit_sequence("Clear Beauty Repair Spots", |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or_else(|| bad("beauty.clearRepairSpots", "clip is gone"))?;
        let e =
            it.effects.iter_mut().find(|e| data::person(e) == Some(name.as_str())).ok_or_else(|| bad("beauty.clearRepairSpots", "person is not assigned"))?;
        e.params.remove("repair_spots");
        Ok(())
    })?;
    Ok(Value::Null)
}

fn model_status(s: &mut Session, _p: &Value) -> Result<Value> {
    let directory = super::beauty_backend::model_dir(s);
    let files: Vec<_> = filmcraft_face_analysis::models::FILES
        .iter()
        .map(|f| json!({"name":f.name,"bytes":f.size,"sha256":f.sha256,"source":f.url,"license":f.license,"licenseUrl":f.license_url}))
        .collect();
    Ok(
        json!({"available":super::beauty_backend::available(),"installed":directory.as_deref().is_some_and(super::beauty_backend::installed),"directory":directory.map(|p|p.to_string_lossy().into_owned()),"files":files,"bytes":filmcraft_face_analysis::models::FILES.iter().map(|f|f.size).sum::<u64>()}),
    )
}

fn download_models(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(all(feature = "face-analysis", not(target_arch = "wasm32")))]
    {
        let directory = super::beauty_backend::model_dir(s).ok_or_else(|| bad("beauty.downloadModels", "application data directory is unavailable"))?;
        if s.jobs.iter().any(|j| j.label == "Download Face Models" && lock(&j.result).is_none()) {
            return Err(bad("beauty.downloadModels", "face model download is already running"));
        }
        let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0).checked_add(1).ok_or_else(|| bad("beauty.downloadModels", "job id limit reached"))?;
        let job = crate::Job { id, label: "Download Face Models".into(), progress: Default::default(), result: Default::default() };
        let progress = job.progress.clone();
        let result = job.result.clone();
        let run = move || {
            let began = web_time::Instant::now();
            let work = || -> std::result::Result<(), String> {
                use sha2::{Digest, Sha256};
                std::fs::create_dir_all(&directory).map_err(|e| format!("cannot create face model directory: {e}"))?;
                for f in filmcraft_face_analysis::models::FILES {
                    let path = directory.join(f.name);
                    if let Ok(metadata) = std::fs::metadata(&path) {
                        let valid = if metadata.is_file() && metadata.len() == f.size {
                            let bytes = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", f.name))?;
                            let hash = format!("{:x}", Sha256::digest(&bytes));
                            hash == f.sha256
                        } else {
                            false
                        };
                        if !valid {
                            let quarantine = directory.join(format!("{}.invalid-{}-{id}", f.name, crate::autosave::unix_now()));
                            std::fs::rename(&path, &quarantine).map_err(|e| format!("cannot preserve invalid model {}: {e}", path.display()))?;
                        }
                    }
                }
                let files: Vec<_> = filmcraft_face_analysis::models::FILES
                    .iter()
                    .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
                    .collect();
                filmcraft_speech::models::download_files(&directory, &files, &mut |done, total, file| {
                    progress.done.store(done, Ordering::Relaxed);
                    progress.total.store(total, Ordering::Relaxed);
                    *lock(&progress.status) = format!("Downloading {file}");
                    !progress.cancel.load(Ordering::Relaxed)
                })
                .map_err(|e| format!("cannot download face models: {e}"))?;
                std::fs::write(directory.join("LICENSE-APACHE.txt"), include_str!("../../../LICENSE-APACHE"))
                    .map_err(|e| format!("cannot write face model license: {e}"))?;
                for f in filmcraft_face_analysis::models::FILES {
                    let attribution = format!(
                        "asset: {}\ntitle: MediaPipe face analysis model\nauthor: Google / The MediaPipe Authors\nsource: {}\nlicense: {}\nlicense-file: LICENSE-APACHE.txt\nadded: {} by FilmCraft\nnotes: Original weights, SHA-256 {}; model card {}\n",
                        f.name,
                        f.url,
                        f.license,
                        download_date(),
                        f.sha256,
                        f.license_url
                    );
                    std::fs::write(directory.join(format!("{}.attribution", f.name)), attribution)
                        .map_err(|e| format!("cannot write face model attribution: {e}"))?;
                }
                super::beauty_backend::analyzer(Some(&directory))?;
                Ok(())
            };
            let complete = work();
            let report = complete.map(|()| filmcraft_export::Report {
                path: directory.to_string_lossy().into_owned(),
                frames: 0,
                seconds: began.elapsed().as_secs_f64(),
                bytes: filmcraft_face_analysis::models::FILES.iter().map(|f| f.size).sum(),
                render_fps: 0.0,
                extra_files: Vec::new(),
            });
            *lock(&progress.status) = match &report {
                Ok(_) => "Face models ready; choose Detect Faces".into(),
                Err(e) => e.clone(),
            };
            if let Err(e) = &report {
                *lock(&progress.error) = Some(e.clone());
            }
            *lock(&result) = Some(report);
            progress.finished.store(true, Ordering::Relaxed);
        };
        let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
        s.jobs.push(job);
        if bool_p(p, "wait").unwrap_or(false) {
            run();
        } else if let Err(e) = std::thread::Builder::new().name("filmcraft-face-model-download".into()).spawn(run) {
            if let Some(j) = s.jobs.iter().find(|j| j.id == id) {
                *lock(&j.result) = Some(Err(e.to_string()));
                j.progress.finished.store(true, Ordering::Relaxed);
            }
            return Err(bad("beauty.downloadModels", format!("cannot start download: {e}")));
        }
        let status = s.jobs.iter().find(|j| j.id == id).map(crate::Job::to_json).unwrap_or(Value::Null);
        if let Some(error) = status.get("result").and_then(|r| r.get("error")).and_then(Value::as_str) {
            return Err(bad("beauty.downloadModels", error));
        }
        Ok(json!({"job":id,"status":status}))
    }
    #[cfg(not(all(feature = "face-analysis", not(target_arch = "wasm32"))))]
    {
        let _ = (s, p);
        Err(bad("beauty.downloadModels", "face model downloads require a desktop build with the face-analysis feature"))
    }
}

#[cfg(all(feature = "face-analysis", not(target_arch = "wasm32")))]
fn download_date() -> String {
    let days = crate::autosave::unix_now().max(0) / 86_400;
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    format!("{year:04}-{month:02}-{day:02}")
}

fn detect(s: &mut Session, p: &Value) -> Result<Value> {
    start(s, p, true)
}
fn track(s: &mut Session, p: &Value) -> Result<Value> {
    start(s, p, false)
}

fn start(s: &mut Session, p: &Value, is_detection: bool) -> Result<Value> {
    let command = if is_detection { "beauty.detect" } else { "beauty.track" };
    let clip = target_clip(s, p)?;
    if s.beauty_jobs.iter().any(|j| j.clip == clip) {
        return Err(bad(command, "face analysis is already running for this clip"));
    }
    let sequence = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = q.find_item(clip).ok_or_else(|| bad(command, "clip is gone"))?;
    require_source_geometry(it)?;
    let it = it.clone();
    let ph = s.playhead().clamp(it.start, (it.end() - Tick(1)).max(it.start));
    let rate = q.settings.frame_rate;
    let frame_ticks = rate.frame_duration();
    if frame_ticks.0 <= 0 || it.duration.0 <= 0 || !it.speed.is_finite() {
        return Err(bad(command, "invalid clip timing"));
    }
    let backward = match str_p(p, "direction").unwrap_or("forward") {
        "forward" => false,
        "backward" => true,
        _ => return Err(bad(command, "direction must be forward or backward")),
    };
    let remaining = if backward { (ph - it.start).0 / frame_ticks.0 } else { (it.end() - Tick(1) - ph).0 / frame_ticks.0 };
    let available = usize::try_from(remaining.max(0)).map_err(|_| bad(command, "clip has too many frames"))?;
    let steps = if is_detection {
        0
    } else {
        let requested = match p.get("frames") {
            Some(v) => {
                usize::try_from(v.as_u64().ok_or_else(|| bad(command, "frames must be a positive integer"))?).map_err(|_| bad(command, "too many frames"))?
            }
            None => available.min(MAX_FRAMES),
        };
        if requested == 0 || requested > MAX_FRAMES {
            return Err(bad(command, "track 1..18000 frames at a time; use a shorter range"));
        }
        requested.min(available)
    };
    if !is_detection && steps == 0 {
        return Err(bad(command, "already at the clip boundary"));
    }
    let name = p.get("person").map(|_| person_param(p)).transpose()?.map(str::to_string);
    let mut targets = Vec::new();
    for e in &it.effects {
        let Some(person) = data::person(e) else { continue };
        if data::confidence_at(e, it.effect_time_at(ph)) < data::MIN_CONFIDENCE {
            if is_detection || name.as_deref().is_none_or(|requested| requested == person) {
                return Err(bad(
                    command,
                    "face identity is unassigned at this frame; manually adjust its region or return to a confident analysed frame first",
                ));
            }
            // An uncertain, unselected peer is not a valid identity association candidate.
            continue;
        }
        let geometry = data::geometry_at(e, it.effect_time_at(ph)).map_err(|e| bad(command, e))?.ok_or_else(|| bad(command, "face geometry is missing"))?;
        targets.push((person.to_string(), geometry));
    }
    if targets.len() > data::MAX_PERSONS {
        return Err(bad(command, "duplicate beauty person targets; remove the duplicate effect"));
    }
    if !is_detection && targets.is_empty() {
        return Err(bad(command, "detect or manually select a face first"));
    }
    if name.as_deref().is_some_and(|n| !targets.iter().any(|(person, _)| person == n)) {
        return Err(bad(command, "the selected person has not been assigned a face"));
    }
    let use_landmarks = is_detection || bool_p(p, "landmarks").unwrap_or(true);
    let analyzer = if use_landmarks { s.face_analyzer.clone() } else { None };
    let model_dir = super::beauty_backend::model_dir(s);
    let source = s.source(it.item).ok_or_else(|| bad(command, "clip media is unavailable; link the original media first"))?;
    let size = filmcraft_render::source_size(&s.project, it.item).ok_or_else(|| bad(command, "clip has no video frame size"))?;
    let scale = (640.0 / size.0.max(size.1).max(1) as f32).min(1.0);
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0).checked_add(1).ok_or_else(|| bad(command, "job id limit reached"))?;
    let job = crate::Job {
        id,
        label: if is_detection { "Detect Beauty Faces".into() } else { "Track Beauty Faces".into() },
        progress: Default::default(),
        result: Default::default(),
    };
    let output: Arc<Mutex<Option<Vec<FaceKey>>>> = Arc::default();
    let out = output.clone();
    let progress = job.progress.clone();
    let result = job.result.clone();
    let original = it.clone();
    let original_item = s.project.item(it.item).cloned();
    let original_media = s.project.resolve_media(it.item).map(|(id, media, range)| (id, media.clone(), range));
    let run = move || {
        let started = web_time::Instant::now();
        let work = || -> std::result::Result<Vec<FaceKey>, String> {
            if progress.cancel.load(Ordering::Relaxed) {
                return Err("face analysis cancelled; no changes applied".into());
            }
            // Loading/parsing a neural graph belongs to the guarded worker too.
            let analyzer = match analyzer.as_ref() {
                Some(a) => Some(a.clone()),
                None if is_detection => Some(super::beauty_backend::analyzer(model_dir.as_deref())?),
                None if use_landmarks && super::beauty_backend::available() && model_dir.as_deref().is_some_and(super::beauty_backend::installed) => {
                    Some(super::beauty_backend::analyzer(model_dir.as_deref())?)
                }
                None => None,
            };
            let count = steps.checked_add(1).and_then(|n| n.checked_mul(2)).ok_or("face analysis is too large")?;
            let mut keys = Vec::new();
            keys.try_reserve_exact(count).map_err(|e| format!("cannot allocate face analysis: {e}"))?;
            progress.total.store((steps + 1) as u64, Ordering::Relaxed);
            let decode = |timeline: Tick| -> std::result::Result<(usize, usize, Vec<u8>), String> {
                let frame = source
                    .video_frame(filmcraft_media::FrameRequest { time: filmcraft_render::video_source_time(&it, timeline, source.info().frame_rate()), scale })
                    .map_err(|e| format!("cannot decode face analysis frame: {e}"))?;
                analysis_rgba(&frame)
            };
            let (w, h, rgba) = decode(ph)?;
            let mut targets = targets;
            if is_detection {
                let analyzer = analyzer.as_ref().ok_or("face analyzer is unavailable")?;
                let mut faces = analyzer.analyze(w, h, &rgba)?;
                validate_observations(&faces)?;
                faces.retain(|f| f.confidence >= data::MIN_CONFIDENCE);
                if faces.is_empty() {
                    return Err("no confident face found; select the face region manually".into());
                }
                if faces.len() > data::MAX_PERSONS {
                    return Err("more than two faces found; assign A/B regions manually".into());
                }
                faces.sort_by(|a, b| a.geometry.centre()[0].total_cmp(&b.geometry.centre()[0]));
                if targets.is_empty() {
                    for (name, face) in ["A", "B"].iter().zip(&faces) {
                        keys.push(FaceKey { name: (*name).into(), time: it.effect_time_at(ph), geometry: face.geometry.clone(), confidence: face.confidence });
                    }
                } else {
                    let matches = match_faces(&targets, &faces);
                    for ((name, geometry), matched) in targets.iter().zip(matches.iter().copied()) {
                        let face = matched.and_then(|i| faces.get(i));
                        keys.push(FaceKey {
                            name: name.clone(),
                            time: it.effect_time_at(ph),
                            geometry: face.map(|f| f.geometry.clone()).unwrap_or_else(|| geometry.clone()),
                            confidence: face.map(|f| f.confidence).unwrap_or(0.0),
                        });
                    }
                    if keys.iter().all(|k| k.confidence < data::MIN_CONFIDENCE) {
                        return Err("face identity is ambiguous; manually reassign A/B".into());
                    }
                    for name in ["A", "B"] {
                        if targets.iter().any(|(n, _)| n == name) {
                            continue;
                        }
                        if let Some((_, face)) = faces.iter().enumerate().find(|(j, _)| !matches.contains(&Some(*j))) {
                            keys.push(FaceKey { name: name.into(), time: it.effect_time_at(ph), geometry: face.geometry.clone(), confidence: face.confidence });
                        }
                    }
                }
                progress.done.store(1, Ordering::Relaxed);
                return Ok(keys);
            }
            for (name, g) in &targets {
                keys.push(FaceKey { name: name.clone(), time: it.effect_time_at(ph), geometry: g.clone(), confidence: 1.0 });
            }
            let mut previous = filmcraft_render::track::Prepared::new(filmcraft_render::track::Gray::from_rgba8(w, h, &rgba));
            let mut active = vec![true; targets.len()];
            let mut previous_size = (w, h);
            for step in 1..=steps {
                if progress.cancel.load(Ordering::Relaxed) {
                    return Err("face analysis cancelled; no tracking changes applied".into());
                }
                let delta = frame_ticks.0.checked_mul(i64::try_from(step).map_err(|_| "frame index overflows")?).ok_or("frame time overflows")?;
                let tick = if backward { ph.0.checked_sub(delta) } else { ph.0.checked_add(delta) }.ok_or("frame time overflows")?;
                let time = Tick(tick);
                let (w, h, rgba) = decode(time)?;
                let next = filmcraft_render::track::Prepared::new(filmcraft_render::track::Gray::from_rgba8(w, h, &rgba));
                let mut predicted = targets.clone();
                for (i, (_, g)) in predicted.iter_mut().enumerate() {
                    if !active.get(i).copied().unwrap_or(false) {
                        continue;
                    }
                    let [x0, y0, x1, y1] = g.bounds;
                    let (pw, ph) = previous_size;
                    let region = [
                        Vec2::new(x0 * pw as f64, y0 * ph as f64),
                        Vec2::new(x1 * pw as f64, y0 * ph as f64),
                        Vec2::new(x1 * pw as f64, y1 * ph as f64),
                        Vec2::new(x0 * pw as f64, y1 * ph as f64),
                    ];
                    if let Some(motion) = filmcraft_render::track::track_step(&previous, &next, &region, filmcraft_project::TrackMethod::PositionScaleRotation)
                    {
                        let transform = |p: [f64; 2]| {
                            let q = motion.transform.apply(Vec2::new(p[0] * pw as f64, p[1] * ph as f64));
                            [q.x / w as f64, q.y / h as f64]
                        };
                        let corners = [transform([x0, y0]), transform([x1, y0]), transform([x1, y1]), transform([x0, y1])];
                        g.bounds = [
                            corners.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
                            corners.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                            corners.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
                            corners.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
                        ];
                        for p in &mut g.landmarks {
                            *p = transform(*p);
                        }
                        if g.validate().is_err() {
                            if let Some((_, previous_geometry)) = targets.get(i) {
                                *g = previous_geometry.clone();
                            }
                            if let Some(a) = active.get_mut(i) {
                                *a = false;
                            }
                        }
                    } else if let Some(a) = active.get_mut(i) {
                        *a = false;
                    }
                }
                let mut confidence = vec![1.0; targets.len()];
                if let Some(analyzer) = analyzer.as_ref() {
                    let faces = analyzer.analyze(w, h, &rgba)?;
                    validate_observations(&faces)?;
                    for (i, matched) in match_faces(&predicted, &faces).into_iter().enumerate() {
                        if !active.get(i).copied().unwrap_or(false) {
                            continue;
                        }
                        if let Some(face) = matched.and_then(|j| faces.get(j)).filter(|f| f.confidence >= data::MIN_CONFIDENCE) {
                            if let Some((_, g)) = predicted.get_mut(i) {
                                *g = face.geometry.clone();
                            }
                            if let Some(c) = confidence.get_mut(i) {
                                *c = face.confidence;
                            }
                        } else if let Some(a) = active.get_mut(i) {
                            *a = false;
                        }
                    }
                }
                for (i, (name, g)) in predicted.iter().enumerate() {
                    let valid = active.get(i).copied().unwrap_or(false);
                    let geometry = if valid { g.clone() } else { targets.get(i).map(|(_, g)| g.clone()).ok_or("tracking target is gone")? };
                    keys.push(FaceKey {
                        name: name.clone(),
                        time: it.effect_time_at(time),
                        geometry,
                        confidence: if valid { confidence.get(i).copied().unwrap_or(0.0) } else { 0.0 },
                    });
                }
                targets = predicted;
                previous = next;
                previous_size = (w, h);
                progress.done.store((step + 1) as u64, Ordering::Relaxed);
                *lock(&progress.status) = format!("Frame {step} of {steps} · {} face(s) tracked", active.iter().filter(|v| **v).count());
            }
            Ok(keys)
        };
        let completed = work().and_then(|mut keys| {
            if progress.cancel.load(Ordering::Relaxed) {
                Err("face analysis cancelled; no changes applied".into())
            } else {
                // Predict both people for identity association even when only A is being edited.
                // Only the requested person's keyframes are committed.
                if !is_detection {
                    keys.retain(|key| name.as_deref().is_none_or(|person| key.name == person));
                }
                Ok(keys)
            }
        });
        let seconds = started.elapsed().as_secs_f64();
        let report = match completed {
            Ok(keys) => {
                let frames = progress.done.load(Ordering::Relaxed);
                *lock(&out) = Some(keys);
                Ok(filmcraft_export::Report {
                    path: String::new(),
                    frames,
                    seconds,
                    bytes: 0,
                    render_fps: frames as f64 / seconds.max(1e-6),
                    extra_files: Vec::new(),
                })
            }
            Err(e) => {
                *lock(&progress.error) = Some(e.clone());
                Err(e)
            }
        };
        *lock(&progress.status) = match &report {
            Ok(_) => "Face analysis complete".into(),
            Err(e) => e.clone(),
        };
        *lock(&result) = Some(report);
        progress.finished.store(true, Ordering::Relaxed);
    };
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    let wait = bool_p(p, "wait").unwrap_or(false) || cfg!(target_arch = "wasm32");
    s.jobs.push(job);
    s.beauty_jobs.push(PendingBeauty { job: id, clip, sequence, original, original_item, original_media, detect: is_detection, keys: output });
    if wait {
        run();
        poll(s);
    } else if let Err(e) = std::thread::Builder::new().name("filmcraft-beauty-analysis".into()).spawn(run) {
        s.beauty_jobs.retain(|j| j.job != id);
        if let Some(job) = s.jobs.iter().find(|j| j.id == id) {
            *lock(&job.result) = Some(Err(e.to_string()));
            job.progress.finished.store(true, Ordering::Relaxed);
        }
        return Err(bad(command, format!("cannot start face analysis: {e}")));
    }
    let status = s.jobs.iter().find(|j| j.id == id).map(crate::Job::to_json).unwrap_or(Value::Null);
    if wait && let Some(error) = status.get("result").and_then(|r| r.get("error")).and_then(Value::as_str) {
        return Err(bad(command, error));
    }
    Ok(json!({"job":id,"status":status}))
}

/// Unique, conservative association. Crossing/occlusion ambiguity disables an affected target
/// rather than moving its settings onto another face. Tracking predicts motion before matching.
pub fn match_faces(targets: &[(String, FaceGeometry)], faces: &[FaceObservation]) -> Vec<Option<usize>> {
    let cost = |g: &FaceGeometry, f: &FaceObservation| {
        if f.confidence < data::MIN_CONFIDENCE {
            return f64::INFINITY;
        }
        let a = g.centre();
        let b = f.geometry.centre();
        let w = (g.bounds[2] - g.bounds[0]).max(0.01);
        let h = (g.bounds[3] - g.bounds[1]).max(0.01);
        let scale = ((f.geometry.bounds[2] - f.geometry.bounds[0]) / w).ln().abs();
        ((a[0] - b[0]) / w).powi(2) + ((a[1] - b[1]) / h).powi(2) + scale * 0.5
    };
    let mut output = vec![None; targets.len().min(data::MAX_PERSONS)];
    let mut choices: Vec<(f64, usize, usize)> = Vec::new();
    for (i, (_, g)) in targets.iter().take(data::MAX_PERSONS).enumerate() {
        let mut scores: Vec<(f64, usize)> = faces.iter().take(8).enumerate().map(|(j, f)| (cost(g, f), j)).collect();
        scores.sort_by(|a, b| a.0.total_cmp(&b.0));
        let Some(&(best, j)) = scores.first() else { continue };
        if best > 1.0 || !best.is_finite() {
            continue;
        }
        if scores.get(1).is_some_and(|(next, _)| next - best < 0.12) {
            continue;
        }
        choices.push((best, i, j));
    }
    choices.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(score, i, j) in &choices {
        // A single visible face between two predicted people is ambiguous in both directions.
        // Choosing the first target would silently put A's settings onto B after an occlusion.
        if choices.iter().any(|(other, target, candidate)| *candidate == j && *target != i && (other - score).abs() < 0.12) {
            continue;
        }
        if output.contains(&Some(j)) {
            continue;
        }
        if let Some(v) = output.get_mut(i) {
            *v = Some(j);
        }
    }
    output
}

pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.beauty_jobs.len() {
        let Some(pending) = s.beauty_jobs.get(i) else { break };
        let finished = s.jobs.iter().find(|j| j.id == pending.job).is_none_or(|j| lock(&j.result).is_some());
        if !finished {
            i += 1;
            continue;
        }
        let pending = s.beauty_jobs.remove(i);
        if let Some(job) = s.jobs.iter().find(|j| j.id == pending.job)
            && job.progress.cancel.load(Ordering::Relaxed)
        {
            *lock(&job.result) = Some(Err("face analysis cancelled; no changes applied".into()));
            continue;
        }
        let keys = lock(&pending.keys).take();
        let Some(keys) = keys else { continue };
        let result = apply_results(s, &pending, &keys);
        if let Err(error) = result {
            if let Some(job) = s.jobs.iter().find(|j| j.id == pending.job) {
                *lock(&job.result) = Some(Err(error.to_string()));
            }
            s.error_toast("beauty", error.to_string());
        } else {
            let lost = keys.iter().any(|k| k.confidence < data::MIN_CONFIDENCE);
            if lost {
                s.toast("Beauty tracking lost a face or found an ambiguous identity; reassign its region before continuing");
            }
        }
    }
}

fn apply_results(s: &mut Session, pending: &PendingBeauty, keys: &[FaceKey]) -> Result<()> {
    if s.project.item(pending.original.item).map(|i| &i.kind) != pending.original_item.as_ref().map(|i| &i.kind) {
        return Err(bad("beauty", "source media changed during analysis; run analysis again"));
    }
    if s.project.resolve_media(pending.original.item) != pending.original_media.as_ref().map(|(id, media, range)| (*id, media, *range)) {
        return Err(bad("beauty", "resolved source media changed during analysis; run analysis again"));
    }
    let def = filmcraft_project::find_effect("beauty").ok_or_else(|| bad("beauty", "beauty is unavailable"))?;
    s.edit("Analyse Beauty Faces", |project, st| {
        let q = project.sequence_mut(pending.sequence).ok_or_else(|| bad("beauty", "analysis sequence was removed"))?;
        let (_, it) = q.find_item_mut(pending.clip).ok_or_else(|| bad("beauty", "analysis clip was removed"))?;
        require_source_geometry(it)?;
        let old = &pending.original;
        if it.item != old.item
            || it.start != old.start
            || it.duration != old.duration
            || it.source_in != old.source_in
            || it.speed != old.speed
            || it.reverse != old.reverse
            || it.frame_hold != old.frame_hold
            || it.hold_filters != old.hold_filters
        {
            return Err(bad("beauty", "clip timing or media changed during analysis; run analysis again"));
        }
        let original: Vec<_> = old.effects.iter().filter(|e| data::person(e).is_some()).map(|e| (data::person(e), e.param(data::FACE_PATH))).collect();
        let current: Vec<_> = it.effects.iter().filter(|e| data::person(e).is_some()).map(|e| (data::person(e), e.param(data::FACE_PATH))).collect();
        if original != current {
            return Err(bad("beauty", "face assignments changed during analysis; run analysis again"));
        }
        for name in ["A", "B"] {
            let face_keys: Vec<_> = keys.iter().filter(|k| k.name == name).collect();
            if face_keys.is_empty() {
                continue;
            }
            let index = match it.effects.iter().position(|e| data::person(e) == Some(name)) {
                Some(i) => i,
                None if pending.detect => {
                    let mut e = def.instance();
                    let first = face_keys.first().ok_or_else(|| bad("beauty", "analysis target is empty"))?;
                    data::assign(&mut e, name, &first.geometry, first.confidence).map_err(|e| bad("beauty", e))?;
                    let pos = it.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
                    it.effects.insert(pos, e);
                    pos
                }
                None => return Err(bad("beauty", "the tracked beauty effect was removed")),
            };
            let e = it.effects.get_mut(index).ok_or_else(|| bad("beauty", "beauty target is gone"))?;
            for key in face_keys {
                key.geometry.validate().map_err(|e| bad("beauty", e))?;
                let path = e.params.entry(data::FACE_PATH.into()).or_insert_with(|| Param::new(ParamValue::Path(key.geometry.to_path())));
                path.put_keyframe(key.time, ParamValue::Path(key.geometry.to_path()));
                let confidence = e.params.entry(data::CONFIDENCE.into()).or_insert_with(|| Param::new(ParamValue::Float(key.confidence)));
                confidence.put_keyframe(key.time, ParamValue::Float(key.confidence));
                if let Some(k) = confidence.keyframes.iter_mut().find(|k| k.time == key.time) {
                    k.interp = Interpolation::Hold;
                }
            }
            if st.active_sequence == Some(pending.sequence) && st.selected_effect.is_none() {
                st.selected_effect = crate::EffectSel::of(pending.clip, &it.effects, index);
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_size_decoder_frames_are_downscaled_before_analysis() {
        let frame = filmcraft_frame::VideoFrame::rgba8(3840, 2160, vec![128; 3840 * 2160 * 4]);
        let (w, h, rgba) = analysis_rgba(&frame).unwrap();
        assert_eq!((w, h), (640, 360));
        assert_eq!(rgba.len(), w * h * 4);
        assert!(rgba.iter().all(|v| *v == 128));
        let oversized = filmcraft_frame::VideoFrame { width: u32::MAX, height: u32::MAX, ..filmcraft_frame::VideoFrame::rgba8(1, 1, vec![0; 4]) };
        assert!(analysis_rgba(&oversized).is_err());
    }

    fn session() -> (Session, ClipId) {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let clip = s.active_sequence().unwrap().video_tracks.iter().flat_map(|t| &t.items).next().unwrap().id;
        let start = s.active_sequence().unwrap().find_item(clip).unwrap().1.start;
        s.state.selection = vec![clip];
        s.execute("playhead.set", json!({"time":start.0})).unwrap();
        (s, clip)
    }

    fn add_person(s: &mut Session, clip: ClipId, name: &str, x: f64) {
        s.execute("beauty.add", json!({"clip":clip.0,"person":name,"bounds":[x,0.1,0.3,0.8]})).unwrap();
    }

    #[test]
    fn independent_person_parameters_selection_and_undo() {
        let (mut s, clip) = session();
        add_person(&mut s, clip, "A", 0.05);
        add_person(&mut s, clip, "B", 0.6);
        let list = s.execute("beauty.list", json!({"clip":clip.0})).unwrap();
        assert_eq!(list["persons"].as_array().unwrap().len(), 2);
        let a = list["persons"].as_array().unwrap().iter().find(|p| p["person"] == "A").unwrap()["effect"].as_u64().unwrap();
        s.execute("effects.setParam", json!({"clip":clip.0,"effect":a,"param":"slimming","value":65})).unwrap();
        let effects = &s.active_sequence().unwrap().find_item(clip).unwrap().1.effects;
        assert_eq!(effects.iter().find(|e| data::person(e) == Some("A")).unwrap().f64_at("slimming", Tick(0)), 65.0);
        assert_eq!(effects.iter().find(|e| data::person(e) == Some("B")).unwrap().f64_at("slimming", Tick(0)), 0.0);
        s.execute("beauty.select", json!({"clip":clip.0,"person":"B"})).unwrap();
        assert_eq!(s.state.selected_effect.as_ref().unwrap().instance, 1);
        s.undo().unwrap();
        let effects = &s.active_sequence().unwrap().find_item(clip).unwrap().1.effects;
        assert_eq!(effects.iter().find(|e| data::person(e) == Some("A")).unwrap().f64_at("slimming", Tick(0)), 0.0);
    }

    #[test]
    fn malformed_commands_never_change_the_project() {
        let (mut s, clip) = session();
        let before = s.project.clone();
        for params in [
            json!({"person":"C","bounds":[0,0,1,1]}),
            json!({"person":"A","bounds":[0,0,-1,1]}),
            json!({"person":"A","bounds":[0,0,1e300,1]}),
            json!({"person":"A","bounds":[0,0]}),
            json!({"person":"A","bounds":[0,null,1,1]}),
        ] {
            assert!(s.execute("beauty.add", params).is_err());
            assert_eq!(s.project, before);
        }
        add_person(&mut s, clip, "A", 0.1);
        let before = s.project.clone();
        assert!(s.execute("beauty.add", json!({"person":"A","bounds":[0,0,1,1]})).is_err());
        for frames in [json!(-1), json!(0), json!(18001), json!(u64::MAX), json!("10"), json!(1.5)] {
            assert!(s.execute("beauty.track", json!({"frames":frames,"wait":true})).is_err());
            assert_eq!(s.project, before);
        }
    }

    #[test]
    fn geometry_changing_effects_require_explicit_disable_before_face_assignment() {
        let (mut s, clip) = session();
        s.execute("effects.apply", json!({"clips":[clip.0],"effect":"horizontal_flip"})).unwrap();
        let before = s.project.clone();
        assert!(s.execute("beauty.add", json!({"person":"A","bounds":[0.1,0.1,0.3,0.8]})).is_err());
        assert!(Arc::ptr_eq(&s.project, &before));
        let sequence = s.state.active_sequence.unwrap();
        for e in &mut Arc::make_mut(&mut s.project).sequence_mut(sequence).unwrap().find_item_mut(clip).unwrap().1.effects {
            if e.effect == "horizontal_flip" {
                e.enabled = false;
            }
        }
        add_person(&mut s, clip, "A", 0.1);
        for e in &mut Arc::make_mut(&mut s.project).sequence_mut(sequence).unwrap().find_item_mut(clip).unwrap().1.effects {
            if e.effect == "horizontal_flip" {
                e.enabled = true;
            }
        }
        let before = s.project.clone();
        assert!(s.execute("beauty.setGeometry", json!({"person":"A","bounds":[0.2,0.2,0.3,0.7]})).is_err());
        assert!(s.execute("beauty.track", json!({"person":"A","frames":1,"wait":true})).is_err());
        assert!(Arc::ptr_eq(&s.project, &before));
    }

    #[test]
    fn matching_keeps_ids_when_the_detector_changes_order_and_refuses_ambiguity() {
        let a = FaceGeometry::from_bounds([0.05, 0.1, 0.35, 0.9]).unwrap();
        let b = FaceGeometry::from_bounds([0.6, 0.1, 0.9, 0.9]).unwrap();
        let targets = vec![("A".into(), a.clone()), ("B".into(), b.clone())];
        let faces = vec![FaceObservation { geometry: b, confidence: 0.9 }, FaceObservation { geometry: a.clone(), confidence: 0.9 }];
        assert_eq!(match_faces(&targets, &faces), vec![Some(1), Some(0)]);
        let duplicate = vec![FaceObservation { geometry: a.clone(), confidence: 0.9 }, FaceObservation { geometry: a, confidence: 0.9 }];
        assert_eq!(match_faces(&targets, &duplicate), vec![None, None]);
        let targets = vec![
            ("A".into(), FaceGeometry::from_bounds([0.3, 0.1, 0.6, 0.9]).unwrap()),
            ("B".into(), FaceGeometry::from_bounds([0.4, 0.1, 0.7, 0.9]).unwrap()),
        ];
        let visible = vec![FaceObservation { geometry: FaceGeometry::from_bounds([0.35, 0.1, 0.65, 0.9]).unwrap(), confidence: 0.9 }];
        assert_eq!(match_faces(&targets, &visible), vec![None, None]);
    }

    struct FixedAnalyzer;
    impl FaceAnalyzer for FixedAnalyzer {
        fn analyze(&self, _w: usize, _h: usize, _rgba: &[u8]) -> std::result::Result<Vec<FaceObservation>, String> {
            Ok(vec![
                FaceObservation { geometry: FaceGeometry::from_bounds([0.05, 0.1, 0.35, 0.9])?, confidence: 0.9 },
                FaceObservation { geometry: FaceGeometry::from_bounds([0.6, 0.1, 0.9, 0.9])?, confidence: 0.9 },
            ])
        }
    }

    #[test]
    fn detection_commits_both_faces_as_one_undo_step_and_has_no_parameters_in_common() {
        let (mut s, clip) = session();
        s.face_analyzer = Some(Arc::new(FixedAnalyzer));
        s.execute("beauty.detect", json!({"clip":clip.0,"wait":true})).unwrap();
        let list = s.execute("beauty.list", json!({})).unwrap();
        assert_eq!(list["persons"].as_array().unwrap().len(), 2);
        s.undo().unwrap();
        assert!(s.active_sequence().unwrap().find_item(clip).unwrap().1.effects.iter().all(|e| data::person(e).is_none()));
        s.redo().unwrap();
        assert_eq!(s.execute("beauty.list", json!({})).unwrap()["persons"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn manually_adjusting_a_face_does_not_move_the_other_face_and_merges_drag_undo() {
        let (mut s, clip) = session();
        add_person(&mut s, clip, "A", 0.05);
        add_person(&mut s, clip, "B", 0.6);
        let before = s.project.clone();
        s.execute("beauty.setGeometry", json!({"person":"A","bounds":[0.15,0.1,0.3,0.8],"merge":"drag","begin":true})).unwrap();
        s.execute("beauty.setGeometry", json!({"person":"A","bounds":[0.25,0.1,0.3,0.8],"merge":"drag"})).unwrap();
        let list = s.execute("beauty.list", json!({})).unwrap();
        let b = list["persons"].as_array().unwrap().iter().find(|p| p["person"] == "B").unwrap();
        assert_eq!(b["geometry"]["bounds"][0], 0.6);
        s.undo().unwrap();
        assert_eq!(s.project, before);
    }

    #[test]
    fn tracking_and_detection_cannot_revive_a_stale_identity_after_a_seek() {
        let (mut s, _) = crate::masks_tests::tracking_session();
        s.face_analyzer = Some(Arc::new(FixedAnalyzer));
        s.execute("beauty.detect", json!({"wait":true})).unwrap();
        s.set_playhead(filmcraft_time::FrameRate::FPS_24.tick_of(10));
        let before = s.project.clone();
        let jobs = s.jobs.len();
        for (command, params) in [
            ("beauty.track", json!({"person":"A","frames":1,"wait":true})),
            ("beauty.track", json!({"frames":1,"wait":true})),
            ("beauty.detect", json!({"wait":true})),
        ] {
            assert!(s.execute(command, params).is_err(), "{command} must require reassignment after an unanalysed seek");
            assert!(Arc::ptr_eq(&s.project, &before));
            assert_eq!(s.jobs.len(), jobs);
        }
    }

    #[test]
    fn manual_reassignment_does_not_activate_unanalysed_gaps_in_either_direction() {
        let rate = filmcraft_time::FrameRate::FPS_24;
        for (destination, gap) in [(40, 30), (10, 15)] {
            let (mut s, clip) = crate::masks_tests::tracking_session();
            s.face_analyzer = Some(Arc::new(FixedAnalyzer));
            s.set_playhead(rate.tick_of(20));
            s.execute("beauty.detect", json!({"wait":true})).unwrap();
            s.set_playhead(rate.tick_of(destination));
            s.execute("beauty.setGeometry", json!({"person":"A","bounds":[0.1,0.1,0.3,0.8]})).unwrap();
            let effects = &s.active_sequence().unwrap().find_item(clip).unwrap().1.effects;
            let a = effects.iter().find(|e| data::person(e) == Some("A")).unwrap();
            let b = effects.iter().find(|e| data::person(e) == Some("B")).unwrap();
            assert_eq!(data::confidence_at(a, rate.tick_of(20)), 0.9);
            assert_eq!(data::confidence_at(a, rate.tick_of(destination)), 1.0);
            assert_eq!(data::confidence_at(a, rate.tick_of(gap)), 0.0);
            assert_eq!(data::confidence_at(b, rate.tick_of(destination)), 0.0);
            assert_eq!(b.param(data::FACE_PATH).unwrap().keyframes.len(), 1);
        }
    }

    #[test]
    fn face_tracking_follows_real_known_motion_and_undo_restores_both_assignments() {
        let (mut s, clip) = crate::masks_tests::tracking_session();
        let rate = filmcraft_time::FrameRate::FPS_24;
        s.set_playhead(rate.tick_of(10));
        let center = crate::masks_tests::truth(10.0).apply(Vec2::ZERO);
        s.execute("beauty.add", json!({"clip":clip.0,"person":"A","bounds":[(center.x-35.0)/320.0,(center.y-35.0)/240.0,70.0/320.0,70.0/240.0]})).unwrap();
        s.execute("beauty.add", json!({"clip":clip.0,"person":"B","bounds":[0.78,0.05,0.19,0.30]})).unwrap();
        let before = s.project.clone();
        s.execute("beauty.track", json!({"person":"A","frames":5,"landmarks":false,"wait":true})).unwrap();
        assert!(s.beauty_jobs.is_empty());
        let effects = &s.active_sequence().unwrap().find_item(clip).unwrap().1.effects;
        let a = effects.iter().find(|e| data::person(e) == Some("A")).unwrap();
        let b = effects.iter().find(|e| data::person(e) == Some("B")).unwrap();
        let tracked = data::geometry_at(a, rate.tick_of(15)).unwrap().unwrap().centre();
        let expected = crate::masks_tests::truth(15.0).apply(Vec2::ZERO);
        assert_eq!(data::confidence_at(a, rate.tick_of(15)), 1.0);
        assert!((tracked[0] * 320.0 - expected.x).abs() < 2.0, "{tracked:?} vs {expected:?}");
        assert!((tracked[1] * 240.0 - expected.y).abs() < 2.0);
        assert!(b.param(data::FACE_PATH).unwrap().keyframes.is_empty(), "tracking A must not edit B");
        s.undo().unwrap();
        assert!(Arc::ptr_eq(&s.project, &before), "undo must restore the original project snapshot, including auto/NaN intrinsic points");
    }

    #[test]
    fn repair_spots_are_person_local_and_can_be_undone() {
        let (mut s, clip) = session();
        add_person(&mut s, clip, "A", 0.05);
        add_person(&mut s, clip, "B", 0.6);
        let before = s.project.clone();
        s.execute("beauty.addRepairSpot", json!({"person":"A","point":[0.2,0.5],"radius":0.03})).unwrap();
        let effects = &s.active_sequence().unwrap().find_item(clip).unwrap().1.effects;
        let a = effects.iter().find(|e| data::person(e) == Some("A")).unwrap();
        let b = effects.iter().find(|e| data::person(e) == Some("B")).unwrap();
        assert_eq!(a.param("repair_spots").unwrap().value.as_path().unwrap().vertices.len(), 1);
        assert!(b.param("repair_spots").is_none());
        let modified = s.project.clone();
        assert!(s.execute("beauty.addRepairSpot", json!({"person":"A","point":[0.9,0.5]})).is_err());
        assert_eq!(s.project, modified);
        s.undo().unwrap();
        assert_eq!(s.project, before);
    }

    #[test]
    fn adding_a_repair_spot_preserves_existing_blemish_animation_and_undo() {
        let (mut s, clip) = session();
        add_person(&mut s, clip, "A", 0.05);
        let now = s.playhead();
        let later = now + Tick::from_seconds_f64(1.0);
        for (time, value) in [(now, 0.0), (later, 80.0)] {
            s.execute("effects.setParam", json!({"clip":clip.0,"effect":"beauty","param":"blemish","value":value,"time":time.0,"keyframe":true})).unwrap();
        }
        let item = s.active_sequence().unwrap().find_item(clip).unwrap().1;
        let (now, later) = (item.effect_time_at(now), item.effect_time_at(later));
        let original = item.effect("beauty").unwrap().param("blemish").unwrap().clone();
        let before = s.project.clone();

        s.execute("beauty.addRepairSpot", json!({"person":"A","point":[0.2,0.5]})).unwrap();

        let item = s.active_sequence().unwrap().find_item(clip).unwrap().1;
        let blemish = item.effect("beauty").unwrap().param("blemish").unwrap();
        assert_eq!(blemish.keyframes.len(), 2, "repair must preserve the existing animation");
        assert_eq!(blemish.value_at(now), ParamValue::Float(50.0));
        assert_eq!(blemish.value_at(later), ParamValue::Float(80.0));
        assert_eq!(blemish.keyframes.last(), original.keyframes.last(), "future keys and their interpolation must be unchanged");
        s.undo().unwrap();
        assert!(Arc::ptr_eq(&s.project, &before), "one undo must restore the spots and the complete original animation");
        let item = s.active_sequence().unwrap().find_item(clip).unwrap().1;
        let restored = item.effect("beauty").unwrap().param("blemish").unwrap();
        assert_eq!(restored.value_at(now), ParamValue::Float(0.0));
        assert_eq!(restored.value_at(later), ParamValue::Float(80.0));
    }

    struct PanickingAnalyzer;
    impl FaceAnalyzer for PanickingAnalyzer {
        fn analyze(&self, _w: usize, _h: usize, _rgba: &[u8]) -> std::result::Result<Vec<FaceObservation>, String> {
            panic!("synthetic worker failure")
        }
    }

    #[test]
    fn failed_worker_finishes_with_error_and_preserves_project() {
        let (mut s, _) = session();
        let before = s.project.clone();
        s.face_analyzer = Some(Arc::new(PanickingAnalyzer));
        assert!(s.execute("beauty.detect", json!({"wait":true})).is_err());
        assert_eq!(s.project, before);
        assert!(s.beauty_jobs.is_empty());
        let job = s.jobs.last().unwrap();
        assert!(job.progress.finished.load(Ordering::Relaxed));
        assert!(lock(&job.result).as_ref().unwrap().is_err());
    }

    #[test]
    fn stale_source_and_changed_hold_filter_mapping_reject_tracking_without_mutating_faces() {
        let (mut s, clip) = session();
        add_person(&mut s, clip, "A", 0.05);
        let original = s.active_sequence().unwrap().find_item(clip).unwrap().1.clone();
        let mut pending = PendingBeauty {
            job: 0,
            clip,
            sequence: s.state.active_sequence.unwrap(),
            original: original.clone(),
            original_item: s.project.item(original.item).cloned(),
            original_media: s.project.resolve_media(original.item).map(|(id, media, range)| (id, media.clone(), range)),
            detect: false,
            keys: Arc::default(),
        };
        let keys = vec![FaceKey { name: "A".into(), time: Tick(0), geometry: FaceGeometry::from_bounds([0.1, 0.1, 0.4, 0.9]).unwrap(), confidence: 1.0 }];
        s.edit_sequence("Changed hold filters", |q, _, _| {
            q.find_item_mut(clip).unwrap().1.hold_filters = true;
            Ok(())
        })
        .unwrap();
        let before = s.project.clone();
        assert!(apply_results(&mut s, &pending, &keys).is_err());
        assert_eq!(s.project, before);
        pending.original = s.active_sequence().unwrap().find_item(clip).unwrap().1.clone();
        pending.original_item = None;
        assert!(apply_results(&mut s, &pending, &keys).is_err());
        assert_eq!(s.project, before);
    }

    #[test]
    fn relinking_a_subclip_parent_invalidates_analysis() {
        let (mut s, clip) = crate::masks_tests::tracking_session();
        let parent = s.active_sequence().unwrap().find_item(clip).unwrap().1.item;
        let sequence = s.state.active_sequence.unwrap();
        let project = Arc::make_mut(&mut s.project);
        let subclip = project.add_item(
            "subclip",
            filmcraft_project::Label::Iris,
            filmcraft_project::ItemKind::Subclip {
                parent,
                range: filmcraft_time::TimeRange::new(Tick::ZERO, filmcraft_time::FrameRate::FPS_24.tick_of(48)),
                restrict_trims: true,
            },
            None,
        );
        project.sequence_mut(sequence).unwrap().find_item_mut(clip).unwrap().1.item = subclip;
        let original = s.active_sequence().unwrap().find_item(clip).unwrap().1.clone();
        let pending = PendingBeauty {
            job: 0,
            clip,
            sequence,
            original: original.clone(),
            original_item: s.project.item(subclip).cloned(),
            original_media: s.project.resolve_media(subclip).map(|(id, media, range)| (id, media.clone(), range)),
            detect: true,
            keys: Arc::default(),
        };
        let filmcraft_project::ItemKind::Media(media) = &mut Arc::make_mut(&mut s.project).item_mut(parent).unwrap().kind else { panic!("media fixture") };
        media.media = filmcraft_project::MediaRef::File { path: "relinked.mp4".into() };
        let before = s.project.clone();
        let keys = [FaceKey { name: "A".into(), time: Tick::ZERO, geometry: FaceGeometry::from_bounds([0.1, 0.1, 0.4, 0.9]).unwrap(), confidence: 1.0 }];
        assert!(apply_results(&mut s, &pending, &keys).is_err());
        assert!(Arc::ptr_eq(&s.project, &before));
    }

    #[test]
    fn cancellation_after_the_worker_publishes_keys_discards_them() {
        let (mut s, clip) = session();
        let before = s.project.clone();
        let original = s.active_sequence().unwrap().find_item(clip).unwrap().1.clone();
        let job = crate::Job { id: 1, label: "finished face analysis".into(), progress: Default::default(), result: Default::default() };
        *lock(&job.result) =
            Some(Ok(filmcraft_export::Report { path: String::new(), frames: 1, seconds: 1.0, bytes: 0, render_fps: 1.0, extra_files: vec![] }));
        job.progress.finished.store(true, Ordering::Relaxed);
        job.progress.cancel.store(true, Ordering::Relaxed);
        let keys = vec![FaceKey { name: "A".into(), time: Tick::ZERO, geometry: FaceGeometry::from_bounds([0.1, 0.1, 0.4, 0.9]).unwrap(), confidence: 1.0 }];
        s.beauty_jobs.push(PendingBeauty {
            job: job.id,
            clip,
            sequence: s.state.active_sequence.unwrap(),
            original: original.clone(),
            original_item: s.project.item(original.item).cloned(),
            original_media: s.project.resolve_media(original.item).map(|(id, media, range)| (id, media.clone(), range)),
            detect: true,
            keys: Arc::new(Mutex::new(Some(keys))),
        });
        s.jobs.push(job);
        poll(&mut s);
        assert!(s.beauty_jobs.is_empty());
        assert!(Arc::ptr_eq(&s.project, &before));
        assert!(lock(&s.jobs[0].result).as_ref().unwrap().is_err());
    }

    #[cfg(all(feature = "face-analysis", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires official local face models and portrait.jpg fixture"]
    fn real_neural_pair_detection_and_render_keep_person_b_unchanged() {
        use filmcraft_media::MediaSource;
        let directory = std::path::PathBuf::from(std::env::var_os("FILMCRAFT_FACE_MODEL_DIR").expect("model fixture directory"));
        let portrait = image::open(directory.join("portrait.jpg")).unwrap().to_rgba8();
        let one = image::imageops::resize(&portrait, 160, 240, image::imageops::FilterType::Triangle);
        let mut pair = image::RgbaImage::new(320, 240);
        image::imageops::replace(&mut pair, &one, 0, 0);
        image::imageops::replace(&mut pair, &one, 160, 0);
        struct StillPair {
            info: filmcraft_media::MediaInfo,
            rgba: Vec<u8>,
        }
        impl MediaSource for StillPair {
            fn info(&self) -> &filmcraft_media::MediaInfo {
                &self.info
            }
            fn video_frame(&self, _: filmcraft_media::FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
                Ok(Arc::new(filmcraft_frame::VideoFrame::rgba8(320, 240, self.rgba.clone())))
            }
            fn audio(&self, _: i64, frames: usize, rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
                Ok(filmcraft_frame::AudioBuffer::silence(rate, 2, frames))
            }
        }
        let (mut s, clip) = crate::masks_tests::tracking_session();
        let item = s.active_sequence().unwrap().find_item(clip).unwrap().1.item;
        let info = s.source(item).unwrap().info().clone();
        s.media.insert(item, Arc::new(StillPair { info, rgba: pair.into_raw() }));
        s.face_analyzer = Some(super::super::beauty_backend::analyzer(Some(&directory)).unwrap());
        s.execute("beauty.detect", json!({"wait":true})).unwrap();
        let persons = s.execute("beauty.list", json!({})).unwrap();
        assert_eq!(persons["persons"].as_array().unwrap().len(), 2);
        let a = persons["persons"].as_array().unwrap().iter().find(|p| p["person"] == "A").unwrap()["effect"].as_u64().unwrap();
        let baseline = s.render_program_working(1.0).unwrap();
        s.execute("effects.setParam", json!({"clip":clip.0,"effect":a,"param":"whitening","value":80})).unwrap();
        let changed = s.render_program_working(1.0).unwrap();
        assert_eq!((changed.w, changed.h), (320, 240));
        let mut left_changed = 0;
        for y in 0..240 {
            for x in 0..320 {
                let i = (y * 320 + x) * 4;
                if x >= 160 {
                    assert_eq!(&changed.px[i..i + 4], &baseline.px[i..i + 4], "person B changed at {x},{y}");
                } else if changed.px[i..i + 4] != baseline.px[i..i + 4] {
                    left_changed += 1;
                }
            }
        }
        assert!(left_changed > 20, "person A's real face must be visibly affected");
        s.execute("beauty.track", json!({"frames":3,"wait":true})).unwrap();
        s.set_playhead(filmcraft_time::FrameRate::FPS_24.tick_of(3));
        let tracked = s.execute("beauty.list", json!({})).unwrap();
        assert!(tracked["persons"].as_array().unwrap().iter().all(|p| p["confidence"].as_f64().unwrap() >= 0.5));
    }
}
