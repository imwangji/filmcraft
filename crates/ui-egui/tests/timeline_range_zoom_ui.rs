//! Right-drag navigation previews before zooming and must never edit clips.
use egui::{Event, Modifiers, PointerButton, Pos2, pos2};
use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_time::Tick;
use filmcraft_ui_egui::{
    FilmcraftApp,
    panels::timeline::{zoom_back, zoom_range, zoom_to_range},
    state::{TimelineView, Tool},
};
use serde_json::json;

struct Driver(Harness<'static, FilmcraftApp>);
impl Driver {
    fn new() -> Self {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let app = FilmcraftApp::new(s);
        let mut d = Self(Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_| app));
        d.frames(4);
        let v = &mut d.0.state_mut().ui.timeline;
        v.pps = 40.0;
        v.target_pps = 40.0;
        v.scroll = 0.0;
        v.target_scroll = 0.0;
        v.fit_pending = false;
        d.frames(4);
        d
    }
    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.0.ctx.clone();
            let mut raw = std::mem::take(self.0.input_mut());
            eframe::App::raw_input_hook(self.0.state_mut(), &ctx, &mut raw);
            *self.0.input_mut() = raw;
            self.0.step();
        }
    }
    fn event(&mut self, e: Event) {
        self.0.input_mut().events.push(e);
        self.frames(1);
    }
    fn right(&mut self, pos: Pos2, pressed: bool) {
        self.event(Event::PointerButton { pos, button: PointerButton::Secondary, pressed, modifiers: Modifiers::NONE });
    }
    fn left_swipe(&mut self) {
        let l = self.0.state_mut().tl.layout.clone().unwrap();
        let from = pos2(l.content.left() + 260.0, l.ruler.center().y);
        let to = from - egui::vec2(180.0, 0.0);
        self.event(Event::PointerMoved(from));
        self.right(from, true);
        self.event(Event::PointerMoved(to));
        self.right(to, false);
        self.frames(80);
    }
}

#[test]
fn right_drag_highlights_then_fits_without_editing() {
    for on_ruler in [false, true] {
        let mut d = Driver::new();
        d.0.state_mut().ui.tool = Tool::Razor;
        let l = d.0.state_mut().tl.layout.clone().unwrap();
        let row = l.rows.iter().find(|r| r.kind == filmcraft_engine::project::TrackKind::Video).unwrap();
        let y = if on_ruler { l.ruler.center().y } else { row.rect.center().y };
        let (from, to) = (pos2(l.content.left() + 80.0, y), pos2(l.content.left() + 280.0, y));
        let before = d.0.state_mut().session.project.clone();
        let playhead = d.0.state_mut().session.playhead();
        d.event(Event::PointerMoved(from));
        d.right(from, true);
        d.event(Event::PointerMoved(to));
        d.frames(2);
        let highlight = d.0.state_mut().auto.query("timeline.zoomRange.selection")[0].rect;
        assert!((highlight[1] - l.ruler.top()).abs() < 0.01);
        assert!((highlight[3] - l.ruler.height()).abs() < 0.01);
        assert!(highlight[1] + highlight[3] <= l.content.top() + 0.01, "highlight must never cover clips");
        assert_eq!(d.0.state_mut().ui.timeline.target_pps, 40.0, "wait for release");
        d.right(to, false);
        d.frames(2);
        let v = &d.0.state_mut().ui.timeline;
        assert!((v.target_scroll - 2.0).abs() < 1e-6, "scroll {}", v.target_scroll);
        assert!((v.target_pps - f64::from(l.content.width()) / 5.0).abs() < 1e-4);
        assert_eq!(*d.0.state_mut().session.project, *before, "Razor must not cut");
        assert_eq!(d.0.state_mut().session.playhead(), playhead);
        assert!(d.0.state_mut().session.state.selection.is_empty());
    }
}

#[test]
fn left_swipes_restore_each_view_then_fit_sequence_without_undoing_edits() {
    let mut d = Driver::new();
    let initial = (d.0.state_mut().ui.timeline.target_pps, d.0.state_mut().ui.timeline.target_scroll);
    zoom_range(d.0.state_mut(), Tick::from_seconds_f64(2.0), Tick::from_seconds_f64(9.0)).unwrap();
    let first = (d.0.state_mut().ui.timeline.target_pps, d.0.state_mut().ui.timeline.target_scroll);
    zoom_range(d.0.state_mut(), Tick::from_seconds_f64(3.0), Tick::from_seconds_f64(4.0)).unwrap();
    d.frames(80);
    d.0.state_mut()
        .session
        .edit("Rename", |p, _| {
            p.name = "Keep this edit".into();
            Ok(())
        })
        .unwrap();
    let project = d.0.state_mut().session.project.clone();
    let revision = d.0.state_mut().session.revision;
    d.left_swipe();
    assert_eq!((d.0.state_mut().ui.timeline.target_pps, d.0.state_mut().ui.timeline.target_scroll), first);
    d.left_swipe();
    assert_eq!((d.0.state_mut().ui.timeline.target_pps, d.0.state_mut().ui.timeline.target_scroll), initial);
    d.left_swipe();
    let app = d.0.state_mut();
    assert_eq!(app.ui.timeline.target_scroll, 0.0);
    assert!(f64::from(app.last_timeline_width) / app.ui.timeline.target_pps >= app.session.active_sequence().unwrap().duration().seconds());
    assert_eq!(app.session.revision, revision);
    assert_eq!(*app.session.project, *project);
}

#[test]
fn histories_are_per_sequence_and_reset_with_the_project() {
    let mut d = Driver::new();
    let first = d.0.state_mut().session.state.active_sequence.unwrap();
    zoom_range(d.0.state_mut(), Tick(100), Tick(1000)).unwrap();
    d.0.state_mut().session.execute("file.newSequence", json!({"name":"Other"})).unwrap();
    zoom_back(d.0.state_mut()).unwrap();
    assert!(d.0.state_mut().ui.timeline.fit_pending, "a different sequence has no history");
    d.0.state_mut().session.state.active_sequence = Some(first);
    zoom_back(d.0.state_mut()).unwrap();
    assert_eq!(d.0.state_mut().ui.timeline.target_pps, 40.0);
    assert_eq!(d.0.state_mut().ui.timeline.target_scroll, 0.0);
    zoom_range(d.0.state_mut(), Tick(100), Tick(1000)).unwrap();
    d.0.state_mut().tl.reset_media_caches();
    zoom_back(d.0.state_mut()).unwrap();
    assert!(d.0.state_mut().ui.timeline.fit_pending);
}

#[test]
fn escape_and_tiny_right_gestures_do_not_zoom() {
    for cancel in [false, true] {
        let mut d = Driver::new();
        let l = d.0.state_mut().tl.layout.clone().unwrap();
        let from = pos2(l.content.left() + 100.0, l.ruler.center().y);
        let to = from + egui::vec2(if cancel { 180.0 } else { 2.0 }, 0.0);
        let before = d.0.state_mut().session.project.clone();
        let playhead = d.0.state_mut().session.playhead();
        d.event(Event::PointerMoved(from));
        d.right(from, true);
        d.event(Event::PointerMoved(to));
        if cancel {
            d.event(Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
        }
        d.right(to, false);
        d.frames(2);
        assert_eq!(d.0.state_mut().ui.timeline.target_pps, 40.0);
        assert_eq!(d.0.state_mut().ui.timeline.target_scroll, 0.0);
        assert_eq!(*d.0.state_mut().session.project, *before);
        assert_eq!(d.0.state_mut().session.playhead(), playhead);
    }
}

#[test]
fn hostile_zoom_parameters_leave_the_view_unchanged() {
    let mut v = TimelineView::default();
    let before = (v.target_pps, v.target_scroll, v.fit_pending);
    for (a, b, w) in [(0, 0, 800.0), (-1, 5, 800.0), (i64::MIN, i64::MAX, 800.0), (0, 5, 0.0), (0, 5, f32::NAN), (0, 5, f32::INFINITY)] {
        assert!(zoom_to_range(&mut v, Tick(a), Tick(b), w).is_err());
        assert_eq!((v.target_pps, v.target_scroll, v.fit_pending), before);
    }
    zoom_to_range(&mut v, Tick(0), Tick(1), 800.0).unwrap();
    assert_eq!(v.target_pps, 24_000.0);
}
