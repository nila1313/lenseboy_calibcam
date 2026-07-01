#!/usr/bin/env bash
set -euo pipefail

cd ~/Desktop/lenseboy_calibcam

OUT_DIR="runs/pipeline1/15_calibcam_all7_ccv_direct"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

VIDEOS=(
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam1_3.ccv"
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam2_3.ccv"
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam3_3.ccv"
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam4_3.ccv"
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam5_3.ccv"
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam6_3.ccv"
  "data/dark_frames-no_common_pose_frame/ccv/20260530_cam7_2.ccv"
)

python -m calibcam \
  --videos "${VIDEOS[@]}" \
  --board "data/dark_frames-no_common_pose_frame/bboboard-v2.npy" \
  --calibration_single \
  --calibration_multi \
  --models pinhole \
  --data_path "$OUT_DIR" \
  2>&1 | tee "${LOG_DIR}/calibcam_all7_ccv_direct.log"
