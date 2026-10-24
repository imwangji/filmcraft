# Per-person beauty

Each selected video clip can have separate **Person A** and **Person B** Beauty effects. Amounts,
enable switches, keyframes, repair spots and geometry belong to that person. Project saves,
undo/redo, previews and exports retain the same target data.

## Using it

Select a video clip and open **Effect Controls**. For automatic setup, apply **Beauty** from
**Effects → Video Effects → Beauty** to show its setup controls. Use **Download Face Models**
once (about 1.5 MB), then **Detect Faces** where both people are visible. Alternatively use
**+ A / + B** beside the **Video** heading to add manual regions, then adjust their boxes on
the Program monitor. Manual boxes use estimated facial positions; automatic detection supplies
mesh landmarks.

Click A/B or a labelled face box to choose whose sliders to edit: Smoothing, Radius, Detail,
Whitening, Rosiness, Blemish Removal, Nasolabial Fold Reduction, Face Slimming and Skin Only.
The enable switch affects only that person. **Repair Spot** selects a local mark explicitly;
Escape finishes the tool. **Clear Repair Spots** removes that person's spots. Spot positions
follow the face box's translation and scale.

Use **Track Both** to analyse both people forward, or a person's forward/backward **Track**
to analyse that person alone. Detection processes the
current frame, not an entire movie. Analysed geometry is keyframed; uncertain/out-of-range
targets are not retouched. Tracking is cancellable and commits in one undo step, at most
18,000 frames per job. The default tracks the next 18,000 frames or the remaining clip,
whichever is shorter; continue a longer clip from the end of that range. After a cut, ambiguous
crossing or lost face, reassign the target. Source/timing changes invalidate pending analysis.

## Implementation and limits

- Google's original BlazeFace and 468-point Face Mesh weights have individually verified
  Apache-2.0 licences. Pure-Rust `tract-tflite` runs them locally; no native interpreter is linked.
  The desktop enables `face-analysis` by default. The web and `--no-default-features` retain
  manual regions. [Backend details](../crates/face-analysis/README.md) include model cards,
  SHA-256 pins, formats, compatibility and inference tests.
- Models live in `<data dir>/models/mediapipe-face/`, overridable with
  `FILMCRAFT_FACE_MODEL_DIR`. The downloader writes licence/attribution sidecars; loading verifies
  size and SHA-256. Playback and inference make no network requests.
- The short-range model suits prominent faces, including two-person interview shots. Distant
  faces, strong profiles, blur and occlusion can fail. Tracking uses motion and conservative
  geometric association, not biometric identity recognition; ambiguous matches require reassignment.
- Automatic blemish treatment reduces isolated redness. Dark marks require explicit spots,
  preserving ordinary moles. Fold reduction softens landmark-localised shadows; it does not
  reconstruct deep wrinkles or promise complete removal of every defect. Skin Only is a colour
  heuristic. No generative face replacement is used.
- Edits feather within the face and protect eyes, lips and nose. An overlapping peer's rectangle
  is excluded even if that peer's beauty is off. This protects the peer but may leave some
  foreground skin untreated; it is not depth segmentation.
- Automatic controls have CPU/WGSL parity. Manual spots and overlapping-peer exclusions use
  CPU fallback. Disjoint A/B effects can use the GPU. Alpha and linear HDR values are retained.
- Intrinsic Motion (position/scale/rotation), colour correction and blur are supported. This
  first version rejects ordinary geometric, temporal and generating effects such as Horizontal
  Flip, Transform, Lens Distortion and Echo: their processed picture no longer matches source
  face coordinates. Disable those effects or bake them into a new source before assigning
  faces. If such an effect is enabled later, per-person beauty and its monitor overlays stop
  until the conflict is removed, rather than applying a person's settings at stale coordinates.

## Commands

Coordinates are normalised source-image coordinates before Motion. Bounds are
`[x,y,width,height]`; person is the stable string `A` or `B`.

| Command | Main parameters |
| --- | --- |
| `beauty.modelStatus` | `{}`; backend, store, files, hashes and licences |
| `beauty.downloadModels` | `{wait?: bool}` |
| `beauty.detect` | `{clip?: id, wait?: bool}` |
| `beauty.add` | `{clip?: id, person, bounds}` |
| `beauty.select` | `{clip?: id, person}` |
| `beauty.list` | `{clip?: id, time?: ticks}`; targets and effect indices |
| `beauty.setGeometry` | `{clip?: id, person, bounds?, landmarks?, time?, merge?, begin?}` |
| `beauty.track` | `{clip?: id, person?, direction?: "forward" or "backward", frames?, landmarks?: bool, wait?}` |
| `beauty.addRepairSpot` | `{clip?: id, person, point: [x,y], radius?: fractionOfFaceWidth}` |
| `beauty.clearRepairSpots` | `{clip?: id, person}` |

Use numeric indices from `beauty.list` with `effects.setParam`, `effects.toggleEnabled` and
existing keyframe commands. Identity, geometry and spots are reserved parameters changed
through `beauty.*`, so a slider command cannot accidentally rebind a person.
`beauty.track` normally refines landmarks with installed models; `landmarks: false` explicitly
uses motion tracking alone, which is also the fallback when no models are installed.

## Validation

Tests cover local repair, fold shadows, deformation, alpha/HDR, hostile inputs, independent
targets, overlapping-peer protection, project round trips, tracking known motion, worker failure
and stale analysis (including a relinked subclip parent). Cancellation after inference cannot
commit already-published keys. GPU comparisons cover each automatic control and combined A/B chains at
multiple resolutions. Headless interaction tests exercise sliders, switches, monitor selection,
dragging/undo and repair spots. Real neural inference uses optional ignored fixtures; no binary
model or test photograph is committed.
