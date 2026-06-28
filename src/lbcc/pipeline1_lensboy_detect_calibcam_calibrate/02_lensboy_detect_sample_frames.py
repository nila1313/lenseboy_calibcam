from pathlib import Path
import csv

import cv2
import numpy as np
import yaml
import lensboy as lb


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_config():
    config_path = Path("configs/pipeline1_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_board(board_path: Path):
    data = np.load(board_path, allow_pickle=True)
    obj = data.item() if getattr(data, "shape", None) == () else data

    if not isinstance(obj, dict):
        raise TypeError(f"Expected board npy to contain dict, got {type(obj)}")

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

    return board, obj


def read_sampled_frames(video_path: Path, sample_step: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_numbers = list(range(0, total_frames, sample_step))

    images = []
    actual_frame_numbers = []

    for frame_idx in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            print(f"⚠️ Could not read frame {frame_idx} from {video_path.name}")
            continue

        images.append(frame)
        actual_frame_numbers.append(frame_idx)

    cap.release()

    return images, actual_frame_numbers, width, height, total_frames


def draw_detection(image, frame_obj, text):
    out = image.copy()

    ids = np.asarray(frame_obj.target_point_indices, dtype=np.int32).reshape(-1, 1)
    corners = np.asarray(frame_obj.detected_points_in_image, dtype=np.float32).reshape(-1, 1, 2)

    cv2.aruco.drawDetectedCornersCharuco(out, corners, ids)

    cv2.putText(
        out,
        text,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    return out


def main():
    config = load_config()

    project_root = Path(config["paths"]["project_root"])
    videos_dir = project_root / config["paths"]["videos_dir"]
    board_path = project_root / config["paths"]["board_file"]

    output_dir = project_root / "runs/pipeline1/02_lensboy_sample_detection"
    detections_dir = output_dir / "detections_npz"
    debug_dir = output_dir / "debug_images"

    detections_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    sample_step = 50

    print_section("Pipeline 1 - Lensboy sample detection")
    print(f"Project root: {project_root}")
    print(f"Videos dir:   {videos_dir}")
    print(f"Board file:   {board_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Sample step:  every {sample_step} frames")

    board, board_dict = load_board(board_path)

    print_section("Board")
    print(board_dict)
    print("Number of ChArUco corners:", len(board.getChessboardCorners()))

    summary_rows = []

    for cam_index, video_name in enumerate(config["videos"]["camera_files"]):
        print_section(f"Camera {cam_index}: {video_name}")

        video_path = videos_dir / video_name
        cam_debug_dir = debug_dir / f"cam{cam_index}"
        cam_debug_dir.mkdir(parents=True, exist_ok=True)

        images, sampled_frame_numbers, width, height, total_frames = read_sampled_frames(
            video_path,
            sample_step,
        )

        print(f"Read sampled images: {len(images)}")
        print(f"Video size: {width}x{height}")
        print(f"Total frames: {total_frames}")
        print(f"Sampled frame numbers: {sampled_frame_numbers}")

        target_points, frames, image_indices = lb.extract_frames_from_charuco(board, images)

        print(f"Lensboy detected frames: {len(frames)} / {len(images)}")

        detected_original_frame_numbers = []
        corner_ids_list = []
        corner_xy_list = []

        for det_i, frame_obj in enumerate(frames):
            sampled_image_index = int(image_indices[det_i])
            original_frame_number = int(sampled_frame_numbers[sampled_image_index])

            ids = np.asarray(frame_obj.target_point_indices, dtype=np.int32)
            xy = np.asarray(frame_obj.detected_points_in_image, dtype=np.float32)

            detected_original_frame_numbers.append(original_frame_number)
            corner_ids_list.append(ids)
            corner_xy_list.append(xy)

            text = f"cam{cam_index} frame {original_frame_number} corners {len(ids)}"
            annotated = draw_detection(images[sampled_image_index], frame_obj, text)

            debug_path = cam_debug_dir / f"cam{cam_index}_frame_{original_frame_number:06d}_corners_{len(ids)}.jpg"
            cv2.imwrite(str(debug_path), annotated)

            print(f"  ✅ frame {original_frame_number}: {len(ids)} corners")

        npz_path = detections_dir / f"cam{cam_index}_lensboy_sample_detections.npz"

        np.savez_compressed(
            npz_path,
            camera_index=np.array(cam_index),
            video_name=np.array(video_name),
            image_width=np.array(width),
            image_height=np.array(height),
            total_frames=np.array(total_frames),
            sample_step=np.array(sample_step),
            sampled_frame_numbers=np.asarray(sampled_frame_numbers, dtype=np.int32),
            detected_frame_numbers=np.asarray(detected_original_frame_numbers, dtype=np.int32),
            target_points=np.asarray(target_points, dtype=np.float32),
            corner_ids_list=np.asarray(corner_ids_list, dtype=object),
            corner_xy_list=np.asarray(corner_xy_list, dtype=object),
            board_dict=np.asarray(board_dict, dtype=object),
        )

        print(f"Saved: {npz_path}")

        summary_rows.append(
            {
                "camera_index": cam_index,
                "video_name": video_name,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "sample_step": sample_step,
                "sampled_images": len(images),
                "detected_frames": len(frames),
                "detection_rate": len(frames) / len(images) if images else 0,
                "detected_frame_numbers": " ".join(map(str, detected_original_frame_numbers)),
                "npz_path": str(npz_path),
            }
        )

    summary_csv = output_dir / "sample_detection_summary.csv"

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print_section("Summary")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved debug images: {debug_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
