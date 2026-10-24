//! Independent A/B Beauty controls and face-region gestures through the control channel.
//! Set FILMCRAFT_UI_SHOTS to render an offscreen screenshot after the interaction test.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::{FilmcraftApp, control::ControlRequest};
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    clip: u64,
    shots: Option<std::path::PathBuf>,
}

impl Driver {
    fn new() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).unwrap();
        let sequence = session.execute("sequence.inspect", json!({})).unwrap();
        let first = &sequence["video"][0]["items"][0];
        let clip = first["clip"].as_u64().unwrap();
        session.execute("timeline.select", json!({"clips": [clip]})).unwrap();
        // The demo fades in from black: park after the fade so screenshots show real media.
        session.execute("playhead.set", json!({"frame": 24})).unwrap();
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let shots = std::env::var_os("FILMCRAFT_UI_SHOTS").map(std::path::PathBuf::from);
        let mut builder = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if shots.is_some() {
            builder = builder.wgpu().with_pixels_per_point(1.0);
        }
        let harness = builder.build_eframe(move |_| app);
        let mut driver = Self { harness, tx, clip, shots };
        driver.frames(4);
        driver.call("ui.panel.show", json!({"panel": "Effect Controls"}));
        driver.frames(3);
        driver
    }

    fn frames(&mut self, count: usize) {
        for _ in 0..count {
            let context = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &context, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        let (request, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(request).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(response) = reply.try_recv() {
                assert_eq!(response["ok"], true, "{method} {params}: {response}");
                return response["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.call("engine.execute", json!({"command": command, "params": params}))
    }

    fn click(&mut self, id: &str) {
        self.call("ui.click", json!({"id": id}));
        self.frames(3);
    }

    fn rect(&mut self, id: &str) -> [f64; 4] {
        let elements = self.call("ui.elements", json!({"prefix": id}));
        let element = elements.as_array().unwrap().iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("missing {id}: {elements}"));
        std::array::from_fn(|index| element["rect"][index].as_f64().unwrap())
    }

    fn drag(&mut self, id: &str, delta: [f64; 2]) {
        let [x, y, w, h] = self.rect(id);
        self.call(
            "ui.drag",
            json!({"from": {"x": x + w * 0.5, "y": y + h * 0.5}, "to": {"x": x + w * 0.5 + delta[0], "y": y + h * 0.5 + delta[1]}, "steps": 6}),
        );
        self.frames(3);
    }

    fn people(&mut self) -> Vec<Value> {
        self.exec("beauty.list", json!({"clip": self.clip}))["persons"].as_array().unwrap().clone()
    }

    fn param(&self, person: &str, parameter: &str) -> f64 {
        let app = self.harness.state();
        let (_, it) = app.session.active_sequence().unwrap().find_item(filmcraft_project::ClipId(self.clip)).unwrap();
        let effect = it.effects.iter().find(|e| filmcraft_project::beauty::person(e) == Some(person)).unwrap();
        effect.f64_at(parameter, filmcraft_time::Tick::ZERO)
    }

    fn repair_spots(&self, person: &str) -> usize {
        let app = self.harness.state();
        let (_, it) = app.session.active_sequence().unwrap().find_item(filmcraft_project::ClipId(self.clip)).unwrap();
        let effect = it.effects.iter().find(|e| filmcraft_project::beauty::person(e) == Some(person)).unwrap();
        match effect.param("repair_spots").map(|p| &p.value) {
            Some(filmcraft_project::ParamValue::Path(path)) => path.vertices.len(),
            _ => 0,
        }
    }

    fn collapse_fixed_effects(&mut self) {
        let app = self.harness.state_mut();
        let (_, it) = app.session.active_sequence().unwrap().find_item(filmcraft_project::ClipId(self.clip)).unwrap();
        for (index, _) in it.effects.iter().enumerate().filter(|(_, e)| e.def().is_some_and(|d| d.intrinsic)) {
            app.ui.collapsed_fx.push(format!("{}:{index}", self.clip));
        }
        self.frames(3);
    }

    fn add_two_people(&mut self) {
        self.click(&format!("beauty.{}.person.A", self.clip));
        self.click(&format!("beauty.{}.person.B", self.clip));
        self.collapse_fixed_effects();
    }

    fn snapshot(&mut self, name: &str) {
        let Some(directory) = self.shots.clone() else { return };
        for _ in 0..200 {
            self.frames(1);
            let app = self.harness.state();
            if app.program_picture().is_some_and(|frame| frame.revision == app.session.revision) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        self.frames(3);
        let image = self.harness.render().expect("render Beauty UI");
        std::fs::create_dir_all(&directory).unwrap();
        image.save(directory.join(format!("{name}.png"))).unwrap();
    }
}

#[test]
fn each_person_has_separate_controls_and_enabling_one_does_not_change_the_other() {
    let mut driver = Driver::new();
    driver.add_two_people();
    assert_eq!(driver.people().len(), 2);
    driver.click(&format!("beauty.{}.person.A", driver.clip));
    driver.drag("effectControls.beauty.A.smoothing.value", [50.0, 0.0]);
    assert!(driver.param("A", "smoothing") > 0.0, "A's visible control changes A");
    assert_eq!(driver.param("B", "smoothing"), 0.0, "B's settings are independent");
    driver.click("effectControls.effect.beauty.A.enabled");
    let people = driver.people();
    assert_eq!(people.iter().find(|p| p["person"] == "A").unwrap()["enabled"], false);
    assert_eq!(people.iter().find(|p| p["person"] == "B").unwrap()["enabled"], true);
    let ids = driver.call("ui.elements", json!({"prefix": "effectControls.beauty."}));
    assert!(
        ids.as_array()
            .unwrap()
            .iter()
            .all(|element| !element["id"].as_str().unwrap().contains("face_path") && !element["id"].as_str().unwrap().contains("face_confidence"))
    );
    driver.snapshot("beauty-two-person-controls");
}

#[test]
fn clicking_a_face_selects_it_and_dragging_its_region_leaves_the_other_person_alone() {
    let mut driver = Driver::new();
    driver.add_two_people();
    let before = driver.people();
    let a_before = before.iter().find(|p| p["person"] == "A").unwrap()["geometry"].clone();
    let b_before = before.iter().find(|p| p["person"] == "B").unwrap()["geometry"].clone();
    driver.click(&format!("program.beauty.{}.A.box", driver.clip));
    let state = driver.exec("state.inspect", json!({}));
    let person_a = before.iter().find(|p| p["person"] == "A").unwrap()["effect"].as_u64().unwrap() as usize;
    let app = driver.harness.state();
    let (_, it) = app.session.active_sequence().unwrap().find_item(filmcraft_project::ClipId(driver.clip)).unwrap();
    assert_eq!(app.session.state.selected_effect.as_ref().unwrap().index(&it.effects), Some(person_a), "face click selects A's effect: {state}");
    let controls = driver.call("ui.elements", json!({"prefix": "effectControls.beauty.A.smoothing.value"}));
    assert!(!controls.as_array().unwrap().is_empty(), "selecting a face also reveals that person's controls");
    driver.drag(&format!("program.beauty.{}.A.box", driver.clip), [18.0, 6.0]);
    let moved = driver.people();
    assert_ne!(moved.iter().find(|p| p["person"] == "A").unwrap()["geometry"], a_before);
    assert_eq!(moved.iter().find(|p| p["person"] == "B").unwrap()["geometry"], b_before);
    driver.exec("edit.undo", json!({}));
    let undone = driver.people();
    assert_eq!(undone.iter().find(|p| p["person"] == "A").unwrap()["geometry"], a_before, "one undo restores the entire region drag");
    driver.snapshot("beauty-face-selection");
}

#[test]
fn spot_repair_applies_only_to_the_selected_face_and_escape_leaves_repair_mode() {
    let mut driver = Driver::new();
    driver.add_two_people();
    driver.click(&format!("beauty.{}.person.A", driver.clip));
    let [x, y, width, height] = driver.rect(&format!("program.beauty.{}.A.box", driver.clip));
    driver.click("effectControls.beauty.A.repairSpot");
    assert_eq!(driver.harness.state().ui.beauty_repair.as_ref().unwrap().person, "A");
    driver.call("ui.click", json!({"x": x + width * 0.5, "y": y + height * 0.5}));
    driver.frames(3);
    assert_eq!(driver.repair_spots("A"), 1);
    assert_eq!(driver.repair_spots("B"), 0);
    driver.call("ui.key", json!({"key": "Escape"}));
    driver.frames(3);
    assert!(driver.harness.state().ui.beauty_repair.is_none());
    driver.click("effectControls.beauty.A.clearRepairSpots");
    assert_eq!(driver.repair_spots("A"), 0);
    assert_eq!(driver.repair_spots("B"), 0);
}
