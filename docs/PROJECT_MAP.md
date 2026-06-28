# lenseboy_calibcam project map

## Goal

Build two structured camera calibration pipelines.

## Pipeline 1

Lensboy is used only for detection.

Steps:
1. Load videos.
2. Detect ChArUco board using Lensboy.
3. Save detection results.
4. Convert detection results to a format Calibcam can use.
5. Run Calibcam single-camera calibration.
6. Run Calibcam multi-camera calibration.
7. Save reprojection-error report.

## Pipeline 2

Lensboy is used for detection and calibration.

Steps:
1. Load videos.
2. Detect ChArUco board using Lensboy.
3. Calibrate with Lensboy.
4. Export Lensboy camera model.
5. Use Calibcam / calibcamlib for reprojection analysis.

## Folder meaning

external/       downloaded GitHub repositories
data/           input videos, board files, and test frames
configs/        YAML config files
src/lbcc/       our own bridge code
runs/           experiment outputs
reports/        final summaries
logs/           terminal logs
docs/           notes and project explanation
