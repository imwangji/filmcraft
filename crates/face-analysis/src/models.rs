//! Pinned original Google model weights; never embedded in FilmCraft.

/// A verified weight file in the offline model directory.
#[derive(Debug, Clone, Copy)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
    pub license: &'static str,
    pub license_url: &'static str,
}

/// Official legacy detector and mesh files verified on 2026-10-23.
/// The SHA-256 pins the content even if the upstream URL changes.
pub const FILES: [ModelFile; 2] = [
    ModelFile {
        name: "face_detection_short_range.tflite",
        url: "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        sha256: "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633",
        size: 229_032,
        license: "Apache-2.0",
        license_url: "https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf",
    },
    ModelFile {
        name: "face_landmark.tflite",
        url: "https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite",
        sha256: "1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814",
        size: 1_242_398,
        license: "Apache-2.0",
        license_url: "https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview",
    },
];
