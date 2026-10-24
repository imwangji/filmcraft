# filmcraft-face-analysis

L2, pure-Rust, offline face analysis for portrait effects. No editor, UI, frame,
or rendering dependencies. `FaceAnalyzer` accepts tightly packed RGB8/RGBA8 and
returns separate detections containing normalized bounds, confidence, and the
original 468 MediaPipe Face Mesh landmark indices. Tracking/person identity and
beauty processing belong to the caller; a detection's vector position is not a
persistent identity.

The default build exposes the interface, validation, and public model catalogue
without an inference dependency. Enable `mediapipe` for CPU inference through
`tract-tflite = 0.23.9` (MIT OR Apache-2.0). No TensorFlow Lite C/C++, native
interpreter, ONNX Runtime, or third-party codec is linked.

## Offline model files

`mediapipe::MediaPipeAnalyzer::from_dir(&Path)` loads the two original files in
`models::FILES`. The editor can download them into its local model store.
`default_analyzer()` is a convenience factory reading `FILMCRAFT_FACE_MODEL_DIR`;
the editor should pass its own model store directory when that override is absent.
Both files are verified by exact byte length and SHA-256 on every load, before
any model parser is called. Changed, truncated, missing, or unknown files return
an actionable error. Loading and inference never contact the network.

| Original weight file | Bytes | SHA-256 |
| --- | ---: | --- |
| `face_detection_short_range.tflite` | 229032 | `3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633` |
| `face_landmark.tflite` | 1242398 | `1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814` |

Source: Google's [official model catalogue](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md),
which links these exact filenames under `storage.googleapis.com/mediapipe-assets/`.
Both models are authored by Google and licensed **Apache-2.0**, explicitly stated
in their individual model cards:

- [BlazeFace Short Range model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf), page 1.
- [Original 468-point Face Mesh model card](https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview), page 2; Google's `https://mediapipe.page.link/facemesh-mc` redirects here. Page 1 specifies 192x192 crops with a 25% margin on each side and 468 XYZ landmarks.

Verified on 2026-10-23. The model files and test photographs are **not** committed,
embedded, bundled, or shipped. If a distribution elects to ship weights, it must
also ship the Apache-2.0 license and asset attribution required by `AGENTS.md`.

## Pipeline and compatibility

1. Aspect-preserving letterbox to 128x128 RGB floats in [-1, 1].
2. BlazeFace SSD inference, 896 fixed-grid anchors, score sigmoid, and IoU NMS.
3. Separate 1.5x square face crops, rotated using each detection's eye direction.
4. Bilinear resize to 192x192 RGB floats in [0, 1], then Face Mesh inference.
5. Face-presence rejection and projection of mesh coordinates back to the source.

The pinned original graphs use constant FLOAT16 `DEQUANTIZE` and per-channel
`PRELU` operations missing from tract's TFLite frontend. `fold.rs` handles these
without approximating the network: it expands finite IEEE-754 half constants to
equivalent float32 constants in a bounded temporary FlatBuffer, and lowers PReLU
to `max(x, 0) + alpha * min(x, 0)` in the typed graph before optimization. The
original model bytes on disk are unchanged. The temporary binary placeholder
used during import is always replaced before a runnable graph is constructed.
Specifications: TensorFlow Lite schema version 3 and the FlatBuffers wire format.

All public image dimensions and lengths are validated (1..=16777216 pixels,
RGB/RGBA only); face count is 1..=8, confidence must be finite in [0, 1]. Crop and
network outputs are checked for finite values. Loading and inference have unwind
guards around the third-party runtime. Graph rewriting caps the model and
expanded constant sizes, uses checked offsets, and re-verifies the FlatBuffer.

This short-range model is intended for prominent faces near the camera, not
crowds, distant people, profiles looking away, or identity recognition. Severe
occlusion and motion blur can lower confidence. The caller should stop applying
an uncertain track and allow reassignment instead of guessing person identity.

## Validation

```text
cargo test -p filmcraft-face-analysis --all-features
cargo clippy -p filmcraft-face-analysis --all-features --all-targets -- -D warnings
```

The real inference test is deliberately ignored in ordinary CI. To run it,
place the pinned originals and Google's official `portrait.jpg` fixture in a
local ignored directory, set `FILMCRAFT_FACE_MODEL_DIR`, then run:

```text
cargo test -p filmcraft-face-analysis --features mediapipe official_models -- --ignored --nocapture
```

It runs the actual two networks on a portrait and a side-by-side composition,
asserting one/two independent faces and 468 landmarks per detection. The fixture
is available at `https://storage.googleapis.com/mediapipe-assets/portrait.jpg`.
The observed confidence was 0.913 for the portrait and 0.930/0.930 for the pair;
the complete load and three model-inference stages took about 0.25s on the
development machine. These are smoke measurements, not video-quality or
real-time performance guarantees. Synthetic tests separately cover hostile input
sizes/NaNs, corrupt weights, truncated FlatBuffers under `catch_unwind`, duplicate
suppression, rotated backprojection, and exact channel-wise PReLU behavior.
