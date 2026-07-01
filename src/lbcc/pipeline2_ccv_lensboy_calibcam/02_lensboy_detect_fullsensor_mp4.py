from pathlib import Path
import csv
import importlib.util

import cv2
import numpy as np
import lensboy as lb


PROJECT_ROOT = Path("/Users/nilamaitrachaity/Desktop/lenseboy_calibcam")

VIDEO_DIR = PROJECT_ROOT / "runs/pipeline2/01_fullsensor_mp4"
OUT_DIR = PROJECT_ROOT / "runs/pipeline2/04_lensboy_fullsensor_detection"
DETECTIONS_DIR = OUT_DIR / "detections_npz"
DEBUG_DIR = OUT_DIR / "debug_images_best"

DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

BOARD_PATH = PROJECT_ROOT / "data/dark_frames-no_common_pose_frame/bboboard-v2.npy"

# Best modes from pipeline2 diagnosis.
PREPROCESS_BY_CAMERA = {
    0: "clahe",
    1: "clahe",
    2: "gamma_0_6",
    3: "gamma_0_6",
    4: "gray",
    5: "gray",
    6: "gamma_0_6",
}

VIDEOS = [
    "cam0_fullsensor_from_ccv.mp4",
    "cam1_fullsensor_from_ccv.mp4",
    "cam2_fullsensor_from_ccv.mp4",
    "cam3_fullsensor_from_ccv.mp4",
    "cam4_fullsensor_from_ccv.mp4",
    "cam5_fullsensor_from_ccv.mp4",
    "cam6_fullsensor_from_ccv.mp4",
]

BATCH_SIZE = 25
MIN_CORNERS_FOR_GOOD = 10


def load_pipeline1_board_loader():
    script_path = PROJECT_ROOT / "src/lbcc/pipeline1_lensboy_detect_calibcam_calibrate/04_lensboy_detect_all_frames.py"
    spec = importlib.util.spec_from_file_location("p1_detect", script_path)
    p1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p1)
    return p1.load_board


def preprocess_frame(frame, mode):
    if frame.ndim == 2:
        gray = frame
    else:
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
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    load_board = load_pipeline1_board_loader()
    board, board_dict = load_board(BOARD_PATH, legacy_pattern=True)

    print("Pipeline2 Step 2: Lensboy detection on full-sensor MP4s")
    print("Input video dir:", VIDEO_DIR)
    print("Output dir:", OUT_DIR)
    print("No offset is applied here.")
    print("Preprocessing per camera:")
    for cam_idx, mode in PREPROCESS_BY_CAMERA.items():
        print(f"  cam{cam_idx}: {mode}")

    summary_rows = []
    per_camera_detected_sets = {}

    for cam_idx, video_name in enumerate(VIDEOS):
        video_path = VIDEO_DIR / video_name
        preprocess_mode = PREPROCESS_BY_CAMERA[cam_idx]

        print("")
        print("=" * 80)
        print(f"cam{cam_idx}: {video_name}")
        print("Video:", video_path)
        print("Preprocess:", preprocess_mode)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detected_frame_numbers = []
        num_corners = []
        corner_ids_list = []
        corner_xy_list = []
        best_debug = []

        frame_batch = []
        frame_number_batch = []

        def process_batch():
            nonlocal frame_batch, frame_number_batch, best_debug

            if not frame_batch:
                return

            target_points, detected_frames, image_indices = lb.extract_frames_from_charuco(
                board,
                frame_batch,
            )

            for det_i, frame_obj in enumerate(detected_frames):
                local_index = int(image_indices[det_i])
                original_frame_idx = int(frame_number_batch[local_index])

                ids = np.asarray(frame_obj.target_point_indices, dtype=np.int32).reshape(-1)
                xy = np.asarray(frame_obj.detected_points_in_image, dtype=np.float32).reshape(-1, 2)

                detected_frame_numbers.append(original_frame_idx)
                num_corners.append(len(ids))
                corner_ids_list.append(ids)
                corner_xy_list.append(xy)

                best_debug.append({
                    "n_corners": len(ids),
                    "frame_idx": original_frame_idx,
                    "image": frame_batch[local_index].copy(),
                    "frame_obj": frame_obj,
                })

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

            if len(frame_batch) >= BATCH_SIZE:
                process_batch()

            frame_idx += 1

        process_batch()
        cap.release()

        detected_count = len(detected_frame_numbers)
        good_count = sum(c >= MIN_CORNERS_FOR_GOOD for c in num_corners)
        max_corners = max(num_corners) if num_corners else 0
        mean_corners = float(np.mean(num_corners)) if num_corners else 0.0

        print(f"Frames: {total_frames}")
        print(f"Detected frames: {detected_count}/{total_frames}")
        print(f"Good frames >= {MIN_CORNERS_FOR_GOOD} corners: {good_count}")
        print(f"Max corners: {max_corners}")
        print(f"Mean corners over detected frames: {mean_corners:.2f}")

        per_camera_detected_sets[cam_idx] = set(detected_frame_numbers)

        npz_path = DETECTIONS_DIR / f"cam{cam_idx}_lensboy_fullsensor_detections.npz"
        np.savez_compressed(
            npz_path,
            camera_index=np.array(cam_idx),
            video_name=np.array(video_name),
            video_path=np.array(str(video_path)),
            width=np.array(width),
            height=np.array(height),
            fps=np.array(fps),
            total_frames=np.array(total_frames),
            legacy_pattern=np.array(True),
            preprocess_mode=np.array(preprocess_mode),
            detected_frame_numbers=np.asarray(detected_frame_numbers, dtype=np.int32),
            num_corners=np.asarray(num_corners, dtype=np.int32),
            corner_ids_list=np.asarray(corner_ids_list, dtype=object),
            corner_xy_list=np.asarray(corner_xy_list, dtype=object),
        )

        print("Saved NPZ:", npz_path)

        # Save only best 10 debug images per camera.
        cam_debug_dir = DEBUG_DIR / f"cam{cam_idx}"
        cam_debug_dir.mkdir(parents=True, exist_ok=True)

        best_debug = sorted(best_debug, key=lambda x: -x["n_corners"])[:10]

        for item in best_debug:
            label = f"cam{cam_idx} f{item['frame_idx']} corners={item['n_corners']} mode={preprocess_mode}"
            annotated = draw_detection(item["image"], item["frame_obj"], label)
            out_path = cam_debug_dir / f"cam{cam_idx}_frame_{item['frame_idx']:06d}_corners_{item['n_corners']}.jpg"
            cv2.imwrite(str(out_path), annotated)

        summary_rows.append({
            "camera_index": cam_idx,
            "video_name": video_name,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "legacy_pattern": True,
            "preprocess_mode": preprocess_mode,
            "detected_frames": detected_count,
            "good_frames_ge_10": good_count,
            "max_corners": max_corners,
            "mean_corners_detected": mean_corners,
            "npz_path": str(npz_path),
        })

    summary_csv = OUT_DIR / "fullsensor_detection_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "camera_index",
                "video_name",
                "width",
                "height",
                "fps",
                "total_frames",
                "legacy_pattern",
                "preprocess_mode",
                "detected_frames",
                "good_frames_ge_10",
                "max_corners",
                "mean_corners_detected",
                "npz_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    all_frame_numbers = sorted(set().union(*per_camera_detected_sets.values()))

    common_rows = []
    for frame_idx in all_frame_numbers:
        cams = [cam_i for cam_i, s in per_camera_detected_sets.items() if frame_idx in s]
        common_rows.append({
            "frame_idx": frame_idx,
            "num_cameras_detected": len(cams),
            "cameras": " ".join(map(str, cams)),
        })

    common_csv = OUT_DIR / "common_detected_frames_summary.csv"
    with open(common_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "num_cameras_detected", "cameras"],
        )
        writer.writeheader()
        writer.writerows(common_rows)

    print("")
    print("DONE")
    print("Summary CSV:", summary_csv)
    print("Common-frame CSV:", common_csv)


if __name__ == "__main__":
    main()
