from pathlib import Path
import csv

import cv2
import numpy as np
import yaml
import lensboy as lb


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_config():
    with open("configs/pipeline1_config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_board(board_path: Path, legacy_pattern: bool):
    data = np.load(board_path, allow_pickle=True)
    obj = data.item() if getattr(data, "shape", None) == () else data

    board_width = int(obj["boardWidth"])
    board_height = int(obj["boardHeight"])
    square_length = float(obj["square_size_real"])
    marker_length = float(obj["square_size_real"]) * float(obj["marker_size"])
    dictionary_type = int(obj["dictionary_type"])

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)

    board = cv2.aruco.CharucoBoard(
        (board_width, board_height),
        square_length,
        marker_length,
        dictionary,
    )

    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(legacy_pattern)

    return board


def read_frames(video_path: Path, frame_indices):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    valid_indices = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            valid_indices.append(frame_idx)

    cap.release()
    return frames, valid_indices


def preprocess_versions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    versions = {}

    versions["raw_bgr"] = frame
    versions["gray"] = gray

    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe_obj.apply(gray)
    versions["clahe"] = clahe_img

    gamma = 0.6
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    versions["gamma_0_6"] = cv2.LUT(gray, table)

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    versions["sharpen"] = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)

    clahe_blur = cv2.GaussianBlur(clahe_img, (0, 0), 3)
    versions["clahe_sharpen"] = cv2.addWeighted(clahe_img, 1.7, clahe_blur, -0.7, 0)

    return versions


def draw_detection(image, frame_obj, label):
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    ids = np.asarray(frame_obj.target_point_indices, dtype=np.int32).reshape(-1, 1)
    corners = np.asarray(frame_obj.detected_points_in_image, dtype=np.float32).reshape(-1, 1, 2)

    cv2.aruco.drawDetectedCornersCharuco(out, corners, ids)

    cv2.putText(
        out,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def main():
    config = load_config()

    project_root = Path(config["paths"]["project_root"])
    videos_dir = project_root / config["paths"]["videos_dir"]
    board_path = project_root / config["paths"]["board_file"]

    output_dir = project_root / "runs/pipeline1/03_targeted_preprocessing_detection"
    debug_dir = output_dir / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)

    target_frames = [
        0, 8, 10, 50, 100, 137, 150, 153, 200, 205, 209, 211, 213,
        250, 300, 350, 400, 408, 413, 419, 420, 450, 454
    ]

    print_section("Pipeline 1 - targeted Lensboy detection")
    print(f"Testing legacyPattern False and True")
    print(f"Target frames: {target_frames}")
    print(f"Output dir: {output_dir}")

    rows = []

    for cam_index, video_name in enumerate(config["videos"]["camera_files"]):
        print_section(f"Camera {cam_index}: {video_name}")

        video_path = videos_dir / video_name
        original_frames, valid_frame_indices = read_frames(video_path, target_frames)

        cam_debug_dir = debug_dir / f"cam{cam_index}"
        cam_debug_dir.mkdir(parents=True, exist_ok=True)

        best = {
            "corners": 0,
            "frame_idx": None,
            "preprocess": None,
            "legacy": None,
        }

        # Build preprocessing batches.
        version_batches = {}
        for frame in original_frames:
            versions = preprocess_versions(frame)
            for version_name, img in versions.items():
                version_batches.setdefault(version_name, []).append(img)

        for legacy_pattern in [False, True]:
            board = load_board(board_path, legacy_pattern)

            for version_name, images in version_batches.items():
                target_points, detected_frames, image_indices = lb.extract_frames_from_charuco(
                    board,
                    images,
                )

                detected_map = {}

                for det_i, frame_obj in enumerate(detected_frames):
                    local_img_index = int(image_indices[det_i])
                    original_frame_idx = int(valid_frame_indices[local_img_index])
                    n_corners = int(len(frame_obj.target_point_indices))

                    detected_map[local_img_index] = n_corners

                    label = (
                        f"cam{cam_index} f{original_frame_idx} "
                        f"{version_name} legacy={legacy_pattern} corners={n_corners}"
                    )

                    annotated = draw_detection(images[local_img_index], frame_obj, label)

                    out_path = (
                        cam_debug_dir
                        / f"cam{cam_index}_frame_{original_frame_idx:06d}_{version_name}_legacy_{legacy_pattern}_corners_{n_corners}.jpg"
                    )
                    cv2.imwrite(str(out_path), annotated)

                    if n_corners > best["corners"]:
                        best = {
                            "corners": n_corners,
                            "frame_idx": original_frame_idx,
                            "preprocess": version_name,
                            "legacy": legacy_pattern,
                        }

                for local_img_index, original_frame_idx in enumerate(valid_frame_indices):
                    n_corners = detected_map.get(local_img_index, 0)
                    rows.append(
                        {
                            "camera_index": cam_index,
                            "video_name": video_name,
                            "frame_idx": original_frame_idx,
                            "preprocess": version_name,
                            "legacy_pattern": legacy_pattern,
                            "detected": n_corners > 0,
                            "num_corners": n_corners,
                        }
                    )

        if best["corners"] == 0:
            print(f"❌ Best cam{cam_index}: no detections")
        else:
            print(
                f"🏆 Best cam{cam_index}: "
                f"frame={best['frame_idx']}, "
                f"preprocess={best['preprocess']}, "
                f"legacy={best['legacy']}, "
                f"corners={best['corners']}"
            )

    summary_csv = output_dir / "targeted_preprocessing_legacy_summary.csv"

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_section("Summary")
    print(f"Saved CSV: {summary_csv}")
    print(f"Saved debug images: {debug_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
