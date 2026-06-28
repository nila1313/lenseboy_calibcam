from pathlib import Path
import csv
import sys

import cv2
import numpy as np
import yaml


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_file(path: Path, label: str) -> bool:
    if path.exists():
        print(f"✅ {label}: {path}")
        return True
    print(f"❌ {label} missing: {path}")
    return False


def main():
    config_path = Path("configs/pipeline1_config.yaml")

    print_section("Pipeline 1 project check")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = Path(config["paths"]["project_root"]).expanduser()
    videos_dir = project_root / config["paths"]["videos_dir"]
    board_file = project_root / config["paths"]["board_file"]
    output_dir = project_root / "runs/pipeline1/00_check_project"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Videos dir:   {videos_dir}")
    print(f"Board file:   {board_file}")
    print(f"Output dir:   {output_dir}")

    print_section("Check board file")

    board_ok = check_file(board_file, "Board file")

    if board_ok:
        try:
            board_data = np.load(board_file, allow_pickle=True)
            print("✅ Board file loaded with NumPy")
            print(f"Board object type: {type(board_data)}")
            print(f"Board shape: {getattr(board_data, 'shape', 'no shape')}")
            print(f"Board dtype: {getattr(board_data, 'dtype', 'no dtype')}")
        except Exception as e:
            print("❌ Board file exists but could not be loaded")
            print(f"Error: {e}")
            board_ok = False

    print_section("Check videos")

    rows = []
    all_videos_ok = True

    camera_files = config["videos"]["camera_files"]

    for cam_index, video_name in enumerate(camera_files):
        video_path = videos_dir / video_name
        exists = video_path.exists()

        row = {
            "camera_index": cam_index,
            "video_name": video_name,
            "video_path": str(video_path),
            "exists": exists,
            "open_ok": False,
            "width": "",
            "height": "",
            "fps": "",
            "frame_count": "",
            "duration_sec": "",
        }

        if not exists:
            print(f"❌ cam{cam_index}: missing {video_path}")
            all_videos_ok = False
            rows.append(row)
            continue

        cap = cv2.VideoCapture(str(video_path))
        open_ok = cap.isOpened()
        row["open_ok"] = open_ok

        if not open_ok:
            print(f"❌ cam{cam_index}: OpenCV cannot open {video_path}")
            all_videos_ok = False
            rows.append(row)
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0.0

        cap.release()

        row["width"] = width
        row["height"] = height
        row["fps"] = fps
        row["frame_count"] = frame_count
        row["duration_sec"] = duration_sec

        print(
            f"✅ cam{cam_index}: {video_name} | "
            f"{width}x{height} | fps={fps:.3f} | frames={frame_count} | "
            f"duration={duration_sec:.2f}s"
        )

        rows.append(row)

    csv_path = output_dir / "input_video_check.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    report_path = output_dir / "input_check_summary.txt"

    unique_sizes = sorted({(r["width"], r["height"]) for r in rows if r["open_ok"]})
    unique_fps = sorted({r["fps"] for r in rows if r["open_ok"]})
    unique_frame_counts = sorted({r["frame_count"] for r in rows if r["open_ok"]})

    with open(report_path, "w") as f:
        f.write("Pipeline 1 input check summary\n")
        f.write("================================\n\n")
        f.write(f"Project root: {project_root}\n")
        f.write(f"Board file: {board_file}\n")
        f.write(f"Board OK: {board_ok}\n\n")
        f.write(f"Videos dir: {videos_dir}\n")
        f.write(f"All videos OK: {all_videos_ok}\n")
        f.write(f"Unique video sizes: {unique_sizes}\n")
        f.write(f"Unique FPS values: {unique_fps}\n")
        f.write(f"Unique frame counts: {unique_frame_counts}\n")
        f.write(f"\nCSV report: {csv_path}\n")

    print_section("Summary")

    print(f"Saved CSV report: {csv_path}")
    print(f"Saved text report: {report_path}")

    if board_ok and all_videos_ok:
        print("✅ Input check passed.")
    else:
        print("⚠️ Input check finished, but something is missing or unreadable.")
        print("Fix the missing files before running detection.")


if __name__ == "__main__":
    main()
