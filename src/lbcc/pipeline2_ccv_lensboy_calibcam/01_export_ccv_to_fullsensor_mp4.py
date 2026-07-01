from pathlib import Path
import csv

import cv2
import numpy as np
from svidreader import filtergraph


PROJECT_ROOT = Path("/Users/nilamaitrachaity/Desktop/lenseboy_calibcam")

CCV_FILES = [
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam1_3.ccv",
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam2_3.ccv",
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam3_3.ccv",
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam4_3.ccv",
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam5_3.ccv",
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam6_3.ccv",
    "data/dark_frames-no_common_pose_frame/ccv/20260530_cam7_2.ccv",
]

OUT_DIR = PROJECT_ROOT / "runs/pipeline2/01_fullsensor_mp4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 10.0


def to_bgr_uint8(frame):
    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame

    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def main():
    metadata_rows = []

    print("Pipeline2 Step 1: Export CCV to full-sensor MP4")
    print("Output:", OUT_DIR)
    print("No crop. No offset. Full CCV frame coordinates.\n")

    for cam_idx, rel_path in enumerate(CCV_FILES):
        ccv_path = PROJECT_ROOT / rel_path
        out_mp4 = OUT_DIR / f"cam{cam_idx}_fullsensor_from_ccv.mp4"

        print("=" * 80)
        print(f"cam{cam_idx}")
        print("Input CCV :", ccv_path)
        print("Output MP4:", out_mp4)

        reader = filtergraph.get_reader(str(ccv_path), backend="iio", cache=False)
        n_frames = int(reader.n_frames)

        first = np.asarray(reader.get_data(0))
        height, width = first.shape[:2]

        print(f"Frames: {n_frames}")
        print(f"Size:   {width} x {height}")
        print(f"First frame dtype={first.dtype}, min/max={int(first.min())}/{int(first.max())}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, FPS, (width, height), True)

        if not writer.isOpened():
            reader.close()
            raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

        for frame_idx in range(n_frames):
            frame = np.asarray(reader.get_data(frame_idx))
            frame_bgr = to_bgr_uint8(frame)
            writer.write(frame_bgr)

            if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
                print(f"  wrote frame {frame_idx + 1}/{n_frames}")

        writer.release()
        reader.close()

        metadata_rows.append({
            "camera_index": cam_idx,
            "input_ccv": str(ccv_path),
            "output_mp4": str(out_mp4),
            "frames": n_frames,
            "width": width,
            "height": height,
            "fps": FPS,
        })

        print("Saved:", out_mp4)

    metadata_csv = OUT_DIR / "fullsensor_mp4_export_metadata.csv"
    with open(metadata_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["camera_index", "input_ccv", "output_mp4", "frames", "width", "height", "fps"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\nDONE")
    print("Metadata:", metadata_csv)


if __name__ == "__main__":
    main()
