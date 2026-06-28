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


def load_board(board_path: Path, legacy_pattern: bool = True):
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

    return board, obj


def preprocess_frame(frame, mode):
    if mode == "raw_bgr":
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "gray":
        return gray

    if mode == "clahe":
        clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe_obj.apply(gray)

    if mode == "gamma_0_6":
        gamma = 0.6
        table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(gray, table)

    if mode == "sharpen":
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.7, blur, -0.7, 0)

    if mode == "clahe_sharpen":
        clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe_obj.apply(gray)
        blur = cv2.GaussianBlur(clahe_img, (0, 0), 3)
        return cv2.addWeighted(clahe_img, 1.7, blur, -0.7, 0)

    raise ValueError(f"Unknown preprocessing mode: {mode}")


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

    output_dir = project_root / "runs/pipeline1/04_lensboy_full_detection"
    detections_dir = output_dir / "detections_npz"
    debug_dir = output_dir / "debug_images_best"
    detections_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    board, board_dict = load_board(board_path, legacy_pattern=True)

    # Chosen from targeted diagnosis.
    preprocess_by_camera = {
        0: "clahe",
        1: "raw_bgr",
        2: "raw_bgr",
        3: "raw_bgr",
        4: "raw_bgr",
        5: "raw_bgr",
        6: "gamma_0_6",
    }

    batch_size = 25
    min_corners_for_good = 8

    print_section("Pipeline 1 - Lensboy full detection")
    print(f"Output dir: {output_dir}")
    print("legacyPattern=True")
    print(f"batch_size={batch_size}")
    print(f"min_corners_for_good={min_corners_for_good}")
    print("Preprocessing per camera:")
    for cam_i, mode in preprocess_by_camera.items():
        print(f"  cam{cam_i}: {mode}")

    summary_rows = []
    per_camera_detected_sets = {}

    for cam_index, video_name in enumerate(config["videos"]["camera_files"]):
        print_section(f"Camera {cam_index}: {video_name}")

        preprocess_mode = preprocess_by_camera[cam_index]
        video_path = videos_dir / video_name

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detected_frame_numbers = []
        corner_ids_list = []
        corner_xy_list = []
        num_corners_list = []

        debug_candidates = []

        frame_batch = []
        frame_number_batch = []

        def process_batch():
            nonlocal frame_batch, frame_number_batch

            if not frame_batch:
                return

            target_points, detected_frames, image_indices = lb.extract_frames_from_charuco(
                board,
                frame_batch,
            )

            for det_i, frame_obj in enumerate(detected_frames):
                local_index = int(image_indices[det_i])
                original_frame_idx = int(frame_number_batch[local_index])

                ids = np.asarray(frame_obj.target_point_indices, dtype=np.int32)
                xy = np.asarray(frame_obj.detected_points_in_image, dtype=np.float32)
                n_corners = int(len(ids))

                detected_frame_numbers.append(original_frame_idx)
                corner_ids_list.append(ids)
                corner_xy_list.append(xy)
                num_corners_list.append(n_corners)

                debug_candidates.append(
                    {
                        "n_corners": n_corners,
                        "frame_idx": original_frame_idx,
                        "image": frame_batch[local_index].copy(),
                        "frame_obj": frame_obj,
                    }
                )

            frame_batch = []
            frame_number_batch = []

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed = preprocess_frame(frame, preprocess_mode)
            frame_batch.append(processed)
            frame_number_batch.append(frame_idx)

            if len(frame_batch) >= batch_size:
                process_batch()

            frame_idx += 1

        process_batch()
        cap.release()

        detected_count = len(detected_frame_numbers)
        good_count = sum(n >= min_corners_for_good for n in num_corners_list)
        max_corners = max(num_corners_list) if num_corners_list else 0
        mean_corners = float(np.mean(num_corners_list)) if num_corners_list else 0.0

        print(f"Video size: {width}x{height}")
        print(f"Frames: {total_frames}")
        print(f"Preprocess: {preprocess_mode}")
        print(f"Detected frames: {detected_count}/{total_frames}")
        print(f"Good frames >= {min_corners_for_good} corners: {good_count}")
        print(f"Max corners: {max_corners}")
        print(f"Mean corners over detected frames: {mean_corners:.2f}")

        per_camera_detected_sets[cam_index] = set(detected_frame_numbers)

        npz_path = detections_dir / f"cam{cam_index}_lensboy_full_detections.npz"

        np.savez_compressed(
            npz_path,
            camera_index=np.array(cam_index),
            video_name=np.array(video_name),
            image_width=np.array(width),
            image_height=np.array(height),
            fps=np.array(fps),
            total_frames=np.array(total_frames),
            legacy_pattern=np.array(True),
            preprocess_mode=np.array(preprocess_mode),
            min_corners_for_good=np.array(min_corners_for_good),
            detected_frame_numbers=np.asarray(detected_frame_numbers, dtype=np.int32),
            num_corners=np.asarray(num_corners_list, dtype=np.int32),
            target_points=np.asarray(board.getChessboardCorners(), dtype=np.float32),
            corner_ids_list=np.asarray(corner_ids_list, dtype=object),
            corner_xy_list=np.asarray(corner_xy_list, dtype=object),
            board_dict=np.asarray(board_dict, dtype=object),
        )

        print(f"Saved detections: {npz_path}")

        # Save top 10 debug images for this camera.
        cam_debug_dir = debug_dir / f"cam{cam_index}"
        cam_debug_dir.mkdir(parents=True, exist_ok=True)

        debug_candidates = sorted(
            debug_candidates,
            key=lambda x: x["n_corners"],
            reverse=True,
        )[:10]

        for item in debug_candidates:
            label = (
                f"cam{cam_index} f{item['frame_idx']} "
                f"{preprocess_mode} corners={item['n_corners']}"
            )
            annotated = draw_detection(item["image"], item["frame_obj"], label)
            out_path = (
                cam_debug_dir
                / f"cam{cam_index}_frame_{item['frame_idx']:06d}_corners_{item['n_corners']}.jpg"
            )
            cv2.imwrite(str(out_path), annotated)

        summary_rows.append(
            {
                "camera_index": cam_index,
                "video_name": video_name,
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "legacy_pattern": True,
                "preprocess_mode": preprocess_mode,
                "detected_frames": detected_count,
                "good_frames_ge_8": good_count,
                "max_corners": max_corners,
                "mean_corners_detected": mean_corners,
                "npz_path": str(npz_path),
            }
        )

    summary_csv = output_dir / "full_detection_summary.csv"

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Common-frame summary across cameras.
    all_frame_numbers = sorted(set().union(*per_camera_detected_sets.values()))
    common_rows = []

    for frame_idx in all_frame_numbers:
        cams = [cam_i for cam_i, s in per_camera_detected_sets.items() if frame_idx in s]
        common_rows.append(
            {
                "frame_idx": frame_idx,
                "num_cameras_detected": len(cams),
                "cameras": " ".join(map(str, cams)),
            }
        )

    common_csv = output_dir / "common_detected_frames_summary.csv"

    with open(common_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "num_cameras_detected", "cameras"],
        )
        writer.writeheader()
        writer.writerows(common_rows)

    print_section("Final summary")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved common-frame CSV: {common_csv}")
    print(f"Saved debug images: {debug_dir}")

    if common_rows:
        best_common = sorted(
            common_rows,
            key=lambda r: r["num_cameras_detected"],
            reverse=True,
        )[:10]

        print("\nTop common detected frames:")
        for r in best_common:
            print(
                f"frame {r['frame_idx']:3d}: "
                f"{r['num_cameras_detected']} cameras -> {r['cameras']}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
