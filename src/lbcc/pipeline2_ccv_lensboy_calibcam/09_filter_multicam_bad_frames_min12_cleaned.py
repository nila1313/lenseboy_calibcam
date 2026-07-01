from pathlib import Path
import csv
import shutil
import yaml
import numpy as np


PROJECT_ROOT = Path("/Users/nilamaitrachaity/Desktop/lenseboy_calibcam")

IN_DIR = PROJECT_ROOT / "runs/pipeline2/10_calibcam_from_fullsensor_lensboy_min12_cleaned"
DIAG_DIR = PROJECT_ROOT / "runs/pipeline2/13_multicam_reprojection_diagnostics_min12_cleaned"
OUT_DIR = PROJECT_ROOT / "runs/pipeline2/14_calibcam_from_fullsensor_lensboy_min12_cleaned_multicam"

PER_FRAME_ERRORS = DIAG_DIR / "per_frame_multicam_errors.csv"

MAX_ERROR_THRESHOLD = 20.0

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_bad_frame_set():
    bad = set()

    with open(PER_FRAME_ERRORS, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cam = int(row["camera_index"])
            frame = int(row["frame_idx"])
            max_error = float(row["max_error_px"])

            if max_error > MAX_ERROR_THRESHOLD:
                bad.add((cam, frame))

    return bad


def main():
    print("=" * 80)
    print("Filter remaining multi-camera bad observations")
    print("=" * 80)

    bad = load_bad_frame_set()

    print("Input:", IN_DIR)
    print("Diagnostics:", PER_FRAME_ERRORS)
    print("Output:", OUT_DIR)
    print("Rule: remove if multi-camera max_error_px >", MAX_ERROR_THRESHOLD)
    print("Bad camera-frame observations:", len(bad))

    print("")
    print("Bad observations:")
    for cam, frame in sorted(bad):
        print(f"  cam{cam} frame {frame}")

    shutil.copy2(IN_DIR / "bboboard-v2.npy", OUT_DIR / "bboboard-v2.npy")

    summary_rows = []

    for cam_idx in range(7):
        in_file = IN_DIR / f"detection_{cam_idx:03d}.yml"
        out_file = OUT_DIR / f"detection_{cam_idx:03d}.yml"

        with open(in_file, "r") as f:
            det = yaml.safe_load(f)

        frame_idxs = np.asarray(det["frame_idxs"], dtype=int)
        marker_coords = np.asarray(det["marker_coords"], dtype=float)

        before_valid = int(np.sum(frame_idxs >= 0))
        removed = 0

        for slot_idx in range(frame_idxs.shape[1]):
            frame_idx = int(frame_idxs[0, slot_idx])

            if frame_idx < 0:
                continue

            if (cam_idx, frame_idx) in bad:
                frame_idxs[0, slot_idx] = -1
                marker_coords[0, slot_idx, :, :] = np.nan
                removed += 1

        after_valid = int(np.sum(frame_idxs >= 0))

        det["frame_idxs"] = frame_idxs.tolist()
        det["marker_coords"] = marker_coords.tolist()

        with open(out_file, "w") as f:
            yaml.safe_dump(det, f, sort_keys=False)

        summary_rows.append({
            "camera_index": cam_idx,
            "before_valid_frames": before_valid,
            "removed_frames": removed,
            "after_valid_frames": after_valid,
            "output_file": str(out_file),
        })

        print("")
        print(f"cam{cam_idx}")
        print("  before:", before_valid)
        print("  removed:", removed)
        print("  after:", after_valid)

    summary_csv = OUT_DIR / "multicam_cleaning_summary.csv"

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "camera_index",
                "before_valid_frames",
                "removed_frames",
                "after_valid_frames",
                "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Summary:", summary_csv)


if __name__ == "__main__":
    main()
