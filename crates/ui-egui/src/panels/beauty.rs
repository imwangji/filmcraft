//! Per-person Beauty controls and editable face regions on the Program monitor.
//! Project mutations always go through the `beauty.*` engine commands.

use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::{ClipId, EffectInstance, TrackItem, beauty::FaceGeometry};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::{FilmcraftApp, state::Tool, theme::Tokens};

const ROW_H: f32 = 22.0;
const MIN_REGION: f64 = 0.01;
const HANDLE_SIZE: f32 = 8.0;
const COORDINATE_WARNING: &str = "Disable geometric effects before per-person Beauty";

fn tr(text: &str) -> &str {
    crate::i18n::t(text)
}

/// Beauty instances have separate automation namespaces. Existing effects keep their ids.
pub(crate) fn automation_effect(e: &EffectInstance, index: usize) -> String {
    if e.effect == "beauty" {
        filmcraft_project::beauty::person(e).map_or_else(|| format!("beauty.instance{index}"), |person| format!("beauty.{person}"))
    } else {
        e.effect.clone()
    }
}

pub(crate) fn label(e: &EffectInstance, fallback: &str) -> String {
    filmcraft_project::beauty::person(e).map_or_else(|| tr(fallback).to_string(), |person| format!("{} · {} {person}", tr(fallback), tr("Person")))
}

pub(crate) fn hidden_param(e: &EffectInstance, param: &str) -> bool {
    e.effect == "beauty" && matches!(param, "person" | "face_path" | "face_confidence")
}

fn reveal_person_controls(app: &mut FilmcraftApp, clip: ClipId, it: &TrackItem, person: &str) {
    for (index, effect) in it.effects.iter().enumerate().filter(|(_, effect)| effect.effect == "beauty") {
        let key = format!("{}:{index}", clip.0);
        app.ui.collapsed_fx.retain(|collapsed| collapsed != &key);
        if filmcraft_project::beauty::person(effect) != Some(person) {
            app.ui.collapsed_fx.push(key);
        }
    }
}

/// Manual entry points share the existing Video heading instead of displacing other tools.
pub(crate) fn clip_header(app: &mut FilmcraftApp, ui: &mut egui::Ui, heading: Rect, clip: ClipId, it: &TrackItem, actions: &mut Vec<(String, Value)>) {
    let compatible = filmcraft_project::beauty::source_coordinate_conflict(&it.effects).is_none();
    let area = Rect::from_min_max(pos2((heading.max.x - 124.0).max(heading.min.x + 70.0), heading.min.y), pos2(heading.max.x - 6.0, heading.max.y));
    let mut child =
        ui.new_child(egui::UiBuilder::new().max_rect(area).id_salt(("beauty-header", clip.0)).layout(egui::Layout::left_to_right(egui::Align::Center)));
    child.label(egui::RichText::new(tr("Beauty")).size(10.5).color(app.tokens.text_dim));
    for person in ["A", "B"] {
        let response = child.add_enabled(compatible, egui::Button::new(format!("+ {person}")).small()).on_disabled_hover_text(tr(COORDINATE_WARNING));
        app.auto.add_response(&format!("beauty.{}.person.{person}", clip.0), &response, &format!("Person {person}"));
        if response.clicked() {
            let bounds = if person == "A" { [0.15, 0.15, 0.28, 0.50] } else { [0.57, 0.15, 0.28, 0.50] };
            actions.push(("beauty.add".into(), json!({"clip": clip.0, "person": person, "bounds": bounds})));
        }
    }
}

/// The selected clip's two independent subjects; manual regions also work without a model.
pub(crate) fn clip_rows(app: &mut FilmcraftApp, ui: &mut egui::Ui, body: Rect, clip: ClipId, it: &TrackItem, actions: &mut Vec<(String, Value)>) {
    if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
        app.ui.beauty_repair = None;
    }
    let compatible = filmcraft_project::beauty::source_coordinate_conflict(&it.effects).is_none();
    if !compatible {
        if app.ui.beauty_repair.as_ref().is_some_and(|target| target.clip == clip.0) {
            app.ui.beauty_repair = None;
        }
        let (row, _) = ui.allocate_exact_size(vec2(body.width(), ROW_H * 2.0), Sense::hover());
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(row.shrink2(vec2(8.0, 0.0))));
        let warning = child.add(egui::Label::new(egui::RichText::new(tr(COORDINATE_WARNING)).color(app.tokens.render_yellow)).wrap());
        app.auto.add_response(&format!("beauty.{}.coordinateConflict", clip.0), &warning, COORDINATE_WARNING);
    }
    let (row, _) = ui.allocate_exact_size(vec2(body.width(), ROW_H + 4.0), Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(row.shrink2(vec2(8.0, 0.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
    let detect = child.add_enabled(compatible, egui::Button::new(tr("Detect Faces")).small());
    app.auto.add_response(&format!("beauty.{}.detect", clip.0), &detect, "Detect Faces");
    if detect.clicked() {
        actions.push(("beauty.detect".into(), json!({"clip": clip.0})));
    }
    for person in ["A", "B"] {
        let found = it.effects.iter().position(|e| filmcraft_project::beauty::person(e) == Some(person));
        let selected =
            found.is_some_and(|index| app.session.state.selected_effect.as_ref().is_some_and(|sel| sel.clip == clip && sel.index(&it.effects) == Some(index)));
        let text = if found.is_some() { format!("{} {person}", tr("Person")) } else { format!("+ {person}") };
        let response = child
            .add_enabled(compatible || found.is_some(), egui::Button::selectable(selected, text))
            .on_hover_text(tr("Select a person, or add a face region manually"));
        app.auto.add_response(&format!("beauty.{}.person.{person}", clip.0), &response, &format!("Person {person}"));
        if response.clicked() {
            let command = if found.is_some() { "beauty.select" } else { "beauty.add" };
            let mut params = json!({"clip": clip.0, "person": person});
            if found.is_none() {
                params["bounds"] = if person == "A" { json!([0.15, 0.15, 0.28, 0.50]) } else { json!([0.57, 0.15, 0.28, 0.50]) };
            }
            actions.push((command.into(), params));
            // The new/selected subject's controls are visible immediately.
            reveal_person_controls(app, clip, it, person);
        }
    }
    let (row, _) = ui.allocate_exact_size(vec2(body.width(), ROW_H), Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(row.shrink2(vec2(8.0, 0.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
    let download = child.small_button(tr("Download Face Models"));
    app.auto.add_response(&format!("beauty.{}.downloadModels", clip.0), &download, "Download Face Models");
    if download.clicked() {
        actions.push(("beauty.downloadModels".into(), json!({})));
    }
    if ["A", "B"].iter().all(|person| it.effects.iter().any(|effect| filmcraft_project::beauty::person(effect) == Some(*person))) {
        let track = child.add_enabled(compatible, egui::Button::new(tr("Track Both")).small());
        app.auto.add_response(&format!("beauty.{}.trackBoth", clip.0), &track, "Track Both");
        if track.clicked() {
            actions.push(("beauty.track".into(), json!({"clip": clip.0, "direction": "forward"})));
        }
    }
}

/// Tracking buttons belong to a person, so their parameters can never affect the other face.
#[allow(clippy::too_many_arguments)]
pub(crate) fn effect_rows(
    app: &mut FilmcraftApp,
    ui: &mut egui::Ui,
    body: Rect,
    clip: ClipId,
    index: usize,
    effect: &EffectInstance,
    mt: Tick,
    actions: &mut Vec<(String, Value)>,
) {
    let Some(person) = filmcraft_project::beauty::person(effect) else { return };
    let compatible = app
        .session
        .active_sequence()
        .and_then(|sequence| sequence.find_item(clip))
        .is_some_and(|(_, item)| filmcraft_project::beauty::source_coordinate_conflict(&item.effects).is_none());
    let auto = format!("effectControls.{}", automation_effect(effect, index));
    let (row, _) = ui.allocate_exact_size(vec2(body.width(), ROW_H), Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(row.shrink2(vec2(26.0, 0.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
    for (direction, text) in [("backward", "◀ Track"), ("forward", "Track ▶")] {
        let response = child.add_enabled(compatible, egui::Button::new(tr(text)).small());
        app.auto.add_response(&format!("{auto}.track.{direction}"), &response, text);
        if response.clicked() {
            actions.push(("beauty.track".into(), json!({"clip": clip.0, "person": person, "direction": direction})));
        }
    }
    let confidence = filmcraft_project::beauty::confidence_at(effect, mt);
    if confidence < filmcraft_project::beauty::MIN_CONFIDENCE {
        child.label(egui::RichText::new(tr("Check face region")).color(app.tokens.render_yellow));
    }
    let (row, _) = ui.allocate_exact_size(vec2(body.width(), ROW_H), Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(row.shrink2(vec2(26.0, 0.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
    let target = crate::state::BeautyRepairTarget { clip: clip.0, person: person.into() };
    let armed = app.ui.beauty_repair.as_ref() == Some(&target);
    let repair = child.add_enabled(compatible, egui::Button::selectable(armed, tr("Repair Spot")));
    app.auto.add_response(&format!("{auto}.repairSpot"), &repair, "Repair Spot");
    if repair.clicked() {
        app.ui.beauty_repair = if armed { None } else { Some(target) };
        if !armed {
            app.ui.status = tr("Click a blemish inside the selected face; Esc finishes").into();
            actions.push(("beauty.select".into(), json!({"clip": clip.0, "person": person})));
        }
    }
    let clear = child.small_button(tr("Clear Repair Spots"));
    app.auto.add_response(&format!("{auto}.clearRepairSpots"), &clear, "Clear Repair Spots");
    if clear.clicked() {
        app.ui.beauty_repair = None;
        actions.push(("beauty.clearRepairSpots".into(), json!({"clip": clip.0, "person": person})));
    }
    let (row, _) = ui.allocate_exact_size(vec2(body.width(), ROW_H), Sense::hover());
    ui.painter().with_clip_rect(row).text(
        pos2(row.min.x + 26.0, row.center().y),
        Align2::LEFT_CENTER,
        tr("Drag the face box or its corners to adjust"),
        Tokens::ui(10.5),
        app.tokens.text_dim,
    );
}

#[derive(Clone)]
struct FaceDrag {
    clip: ClipId,
    person: String,
    geometry: FaceGeometry,
    inverse: Affine,
    origin: Vec2,
    corner: Option<usize>,
    source_size: (u32, u32),
}

fn source_to_screen(app: &FilmcraftApp, it: &TrackItem, mt: Tick, picture: Rect, frame: (u32, u32)) -> Option<(Affine, (u32, u32))> {
    if frame.0 == 0 || frame.1 == 0 || !picture.is_positive() {
        return None;
    }
    let seq = app.session.active_sequence()?;
    let size = filmcraft_render::source_size(&app.session.project, it.item)?;
    if size.0 == 0 || size.1 == 0 {
        return None;
    }
    let motion = filmcraft_render::motion_matrix(seq, it, size, filmcraft_render::source_par(&app.session.project, it.item), mt);
    let view = Affine::translate(f64::from(picture.min.x), f64::from(picture.min.y))
        .then_apply(&Affine::scale(f64::from(picture.width()) / f64::from(frame.0), f64::from(picture.height()) / f64::from(frame.1)));
    Some((view.then_apply(&motion), size))
}

fn screen_point(matrix: &Affine, point: [f64; 2], size: (u32, u32)) -> Pos2 {
    let point = matrix.apply(Vec2::new(point[0] * f64::from(size.0), point[1] * f64::from(size.1)));
    pos2(point.x as f32, point.y as f32)
}

fn corners(bounds: [f64; 4]) -> [[f64; 2]; 4] {
    let [x0, y0, x1, y1] = bounds;
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn adjusted_geometry(drag: &FaceDrag, pointer: Pos2) -> FaceGeometry {
    let current = drag.inverse.apply(Vec2::new(f64::from(pointer.x), f64::from(pointer.y)));
    let delta = [(current.x - drag.origin.x) / f64::from(drag.source_size.0), (current.y - drag.origin.y) / f64::from(drag.source_size.1)];
    let [x0, y0, x1, y1] = drag.geometry.bounds;
    let mut bounds = drag.geometry.bounds;
    match drag.corner {
        None => {
            let dx = delta[0].clamp(-0.5 - x0, 1.5 - x1);
            let dy = delta[1].clamp(-0.5 - y0, 1.5 - y1);
            bounds = [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
        }
        Some(corner) => {
            if corner == 0 || corner == 3 {
                bounds[0] = (x0 + delta[0]).clamp(-0.5, (x1 - MIN_REGION).max(-0.5));
            } else {
                bounds[2] = (x1 + delta[0]).clamp((x0 + MIN_REGION).min(1.5), 1.5);
            }
            if corner < 2 {
                bounds[1] = (y0 + delta[1]).clamp(-0.5, (y1 - MIN_REGION).max(-0.5));
            } else {
                bounds[3] = (y1 + delta[1]).clamp((y0 + MIN_REGION).min(1.5), 1.5);
            }
        }
    }
    let mut geometry = drag.geometry.clone();
    geometry.bounds = bounds;
    for point in &mut geometry.landmarks {
        point[0] = bounds[0] + (point[0] - x0) * (bounds[2] - bounds[0]) / (x1 - x0).max(MIN_REGION);
        point[1] = bounds[1] + (point[1] - y0) * (bounds[3] - bounds[1]) / (y1 - y0).max(MIN_REGION);
    }
    geometry
}

/// Both subjects are selectable; only the active subject's box has resize handles.
pub fn monitor_overlay(app: &mut FilmcraftApp, ui: &mut egui::Ui, picture: Rect, frame: (u32, u32)) {
    if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
        app.ui.beauty_repair = None;
    }
    if app.ui.tool != Tool::Selection || app.ui.mask_pen.is_some() || app.ui.eyedropper.is_some() || app.session.state.selected_mask.is_some() {
        app.ui.beauty_repair = None;
        return;
    }
    let Some(selected) = app.session.state.selected_effect.clone().filter(|selected| selected.effect == "beauty") else {
        app.ui.beauty_repair = None;
        return;
    };
    if !app.session.state.selection.contains(&selected.clip) {
        return;
    }
    let Some(seq) = app.session.active_sequence() else { return };
    let Some((_, it)) = seq.find_item(selected.clip) else { return };
    let it = it.clone();
    if filmcraft_project::beauty::source_coordinate_conflict(&it.effects).is_some() {
        app.ui.beauty_repair = None;
        return;
    }
    let ph = app.session.playhead();
    if ph < it.start || ph >= it.end() {
        return;
    }
    let mt = it.effect_time_at(ph);
    let Some((to_screen, size)) = source_to_screen(app, &it, mt, picture, frame) else { return };
    let Some(inverse) = to_screen.inverse() else { return };
    let active = selected.index(&it.effects);
    let repair = app.ui.beauty_repair.clone().filter(|target| {
        target.clip == selected.clip.0
            && active.and_then(|index| it.effects.get(index)).is_some_and(|effect| filmcraft_project::beauty::person(effect) == Some(target.person.as_str()))
    });
    if app.ui.beauty_repair.is_some() && repair.is_none() {
        app.ui.beauty_repair = None;
    }
    let mut actions = Vec::new();
    for (index, effect) in it.effects.iter().enumerate() {
        let Some(person) = filmcraft_project::beauty::person(effect) else { continue };
        let Ok(Some(geometry)) = filmcraft_project::beauty::geometry_at(effect, mt) else { continue };
        let active = active == Some(index);
        let confident = filmcraft_project::beauty::confidence_at(effect, mt) >= filmcraft_project::beauty::MIN_CONFIDENCE;
        let color = if !confident {
            app.tokens.render_yellow
        } else if person == "A" {
            Color32::from_rgb(90, 180, 255)
        } else {
            Color32::from_rgb(225, 155, 255)
        };
        let points = corners(geometry.bounds).map(|point| screen_point(&to_screen, point, size));
        let painter = ui.painter().with_clip_rect(picture);
        let mut outline = points.to_vec();
        outline.push(points[0]);
        painter.add(egui::Shape::line(outline, Stroke::new(if active { 2.0 } else { 1.0 }, color)));
        let label = format!("{} {person}", tr("Person"));
        let label_at = points[0] + vec2(3.0, 3.0);
        painter.rect_filled(Rect::from_min_size(label_at, vec2(64.0, 19.0)), 3.0, Color32::from_black_alpha(190));
        painter.text(label_at + vec2(5.0, 9.0), Align2::LEFT_CENTER, label, Tokens::semibold(11.0), color);
        if active
            && repair.is_some()
            && let Some(filmcraft_project::ParamValue::Path(spots)) = effect.param("repair_spots").map(|param| &param.value)
        {
            let [x0, y0, x1, y1] = geometry.bounds;
            for spot in spots.vertices.iter().take(64).filter(|spot| spot.p.x.is_finite() && spot.p.y.is_finite()) {
                let at = screen_point(&to_screen, [x0 + spot.p.x * (x1 - x0), y0 + spot.p.y * (y1 - y0)], size);
                painter.circle_stroke(at, 4.0, Stroke::new(1.0, color));
            }
        }
        let body = Rect::from_points(&points).intersect(picture);
        if !body.is_positive() {
            continue;
        }
        let id = egui::Id::new(("beauty-face", selected.clip.0, person));
        let response = ui.interact(
            body,
            id,
            if repair.is_some() {
                Sense::hover()
            } else if active {
                Sense::click_and_drag()
            } else {
                Sense::click()
            },
        );
        let auto = format!("program.beauty.{}.{}", selected.clip.0, person);
        app.auto.add_response(&format!("{auto}.box"), &response, &format!("Person {person}"));
        if response.clicked() {
            reveal_person_controls(app, selected.clip, &it, person);
            actions.push(("beauty.select", json!({"clip": selected.clip.0, "person": person})));
        }
        let mut gesture = response;
        let mut grabbed_corner = None;
        if active && repair.is_none() {
            for (corner, point) in points.iter().enumerate() {
                let rect = Rect::from_center_size(*point, vec2(HANDLE_SIZE, HANDLE_SIZE)).intersect(picture);
                if !rect.is_positive() {
                    continue;
                }
                painter.rect_filled(rect, 1.0, color);
                painter.rect_stroke(rect, 1.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Inside);
                let response = ui.interact(rect.expand(3.0).intersect(picture), id.with(corner), Sense::drag());
                app.auto.add_response(&format!("{auto}.handle.{corner}"), &response, "Resize face region");
                if response.drag_started() || response.dragged() || response.drag_stopped() {
                    gesture = response;
                    grabbed_corner = Some(corner);
                }
            }
        }
        let drag_id = id.with("gesture");
        if active && gesture.drag_started_by(egui::PointerButton::Primary) {
            let pointer = ui.input(|input| input.pointer.press_origin()).or(gesture.interact_pointer_pos());
            if let Some(pointer) = pointer {
                let origin = inverse.apply(Vec2::new(f64::from(pointer.x), f64::from(pointer.y)));
                ui.data_mut(|data| {
                    data.insert_temp(
                        drag_id,
                        FaceDrag { clip: selected.clip, person: person.into(), geometry, inverse, origin, corner: grabbed_corner, source_size: size },
                    )
                });
            }
        }
        if active
            && gesture.dragged_by(egui::PointerButton::Primary)
            && let Some(drag) = ui.data(|data| data.get_temp::<FaceDrag>(drag_id))
            && let Some(pointer) = gesture.interact_pointer_pos()
        {
            let geometry = adjusted_geometry(&drag, pointer);
            let [x0, y0, x1, y1] = geometry.bounds;
            actions.push((
                "beauty.setGeometry",
                json!({"clip": drag.clip.0, "person": drag.person, "bounds": [x0, y0, x1 - x0, y1 - y0], "landmarks": geometry.landmarks,
                    "merge": format!("beauty-region-{}-{person}", selected.clip.0), "begin": gesture.drag_started()}),
            ));
        }
        if gesture.drag_stopped() {
            ui.data_mut(|data| data.remove::<FaceDrag>(drag_id));
        }
    }
    if let Some(target) = repair {
        let response = ui.interact(picture, egui::Id::new(("beauty-repair", target.clip, &target.person)), Sense::click());
        app.auto.add_response("program.beauty.repairSpot", &response, "Click a blemish to repair");
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let point = inverse.apply(Vec2::new(f64::from(pointer.x), f64::from(pointer.y)));
            let point = [point.x / f64::from(size.0), point.y / f64::from(size.1)];
            let geometry = active.and_then(|index| it.effects.get(index)).and_then(|effect| filmcraft_project::beauty::geometry_at(effect, mt).ok().flatten());
            if geometry.is_some_and(|geometry| geometry.contains(point)) {
                actions.push(("beauty.addRepairSpot", json!({"clip": target.clip, "person": target.person, "point": point})));
            } else {
                app.ui.status = tr("Choose a blemish inside the selected face region").into();
            }
        }
    }
    for (command, params) in actions {
        if let Err(error) = app.session.execute(command, params) {
            app.ui.status = error.to_string();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partly_offscreen_region_wider_than_the_image_can_be_dragged_without_panicking() {
        let geometry = FaceGeometry::from_bounds([-0.3, 0.1, 1.3, 0.9]).unwrap();
        let drag =
            FaceDrag { clip: ClipId(1), person: "A".into(), geometry, inverse: Affine::IDENTITY, origin: Vec2::ZERO, corner: None, source_size: (100, 100) };
        let moved = std::panic::catch_unwind(|| adjusted_geometry(&drag, pos2(1000.0, -1000.0))).unwrap();
        moved.validate().unwrap();
        assert!((moved.bounds[2] - moved.bounds[0] - 1.6).abs() < 1e-9);
        assert!(moved.bounds[2] <= 1.5 && moved.bounds[1] >= -0.5);
    }
}
