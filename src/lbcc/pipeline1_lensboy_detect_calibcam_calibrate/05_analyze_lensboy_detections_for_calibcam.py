from pathlib import Path
import csv
import numpy as np
import yaml


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_config():
    with open("configs/pipeline1_config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    project_root = Path(config["paths"]["project_root"])

    det_dir = project_root / "runs/pipeline1/04_lensboy_full_detection/detections_npz"
    out_dir = project_root / "runs/pipeline1/05_calibcam_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)

    min_corners = 8

    print_section("Pipeline 1 - analyze Lensboy detections for Calibcam")
    print(f"Detection dir: {det_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Min corners per camera per frame: {min_corners}")

    per_camera = {}

    for cam_index in range(len(config["videos"]["camera_files"])):
        npz_path = det_dir / f"cam{cam_index}_lensboy_full_detections.npz"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing detection file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        frame_numbers = data["detected_frame_numbers"].astype(int)
        num_corners = data["num_corners"].astype(int)

        good_mask = num_corners >= min_corners

        good_frames = frame_numbers[good_mask]
        good_corners = num_corners[good_mask]

        frame_to_corners = {
            int(f): int(c)
            for f, c in zip(good_frames, good_corners)
        }

        per_camera[cam_index] = frame_to_corners

        print(
            f"cam{cam_index}: "
            f"detected={len(frame_numbers)}, "
            f"good_ge_{min_corners}={len(good_frames)}, "
            f"max_corners={int(num_corners.max()) if len(num_corners) else 0}"
        )

    all_good_frames = sorted(set().union(*[set(v.keys()) for v in per_camera.values()]))

    rows = []

    for frame_idx in all_good_frames:
        cams = []
        corner_counts = []

        for cam_index, frame_map in per_camera.items():
            if frame_idx in frame_map:
                cams.append(cam_index)
                corner_counts.append(frame_map[frame_idx])

        rows.append(
            {
                "frame_idx": frame_idx,
                "num_good_cameras": len(cams),
                "cameras": " ".join(map(str, cams)),
                "corner_counts": " ".join(map(str, corner_counts)),
                "total_corners": sum(corner_counts),
                "min_corners_in_frame": min(corner_counts) if corner_counts else 0,
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["num_good_cameras"],
            r["total_corners"],
            r["min_corners_in_frame"],
        ),
        reverse=True,
    )

    csv_path = out_dir / "good_common_frames_min8.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "num_good_cameras",
                "cameras",
                "corner_counts",
                "total_corners",
                "min_corners_in_frame",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    print_section("Top good common frames")

    for r in rows_sorted[:30]:
        print(
            f"frame {r['frame_idx']:3d}: "
            f"{r['num_good_cameras']} cams -> {r['cameras']} | "
            f"corners: {r['corner_counts']} | "
            f"total={r['total_corners']}"
        )

    print_section("Counts by number of good cameras")

    counts = {}
    for r in rows:
        counts[r["num_good_cameras"]] = counts.get(r["num_good_cameras"], 0) + 1

    for ncam in sorted(counts.keys(), reverse=True):
        print(f"{ncam} good cameras: {counts[ncam]} frames")

    print_section("Saved")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
