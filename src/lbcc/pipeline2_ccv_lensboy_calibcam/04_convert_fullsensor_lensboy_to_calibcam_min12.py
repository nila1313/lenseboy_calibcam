from pathlib import Path
import shutil
import numpy as np
import yaml

from calibcamlib import Detections


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

    lensboy_det_dir = project_root / "runs/pipeline2/04_lensboy_fullsensor_detection/detections_npz"

    output_dir = project_root / "runs/pipeline2/07_calibcam_from_fullsensor_lensboy_min12"
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_out = output_dir / "detection.yml"
    board_out = output_dir / "bboboard-v2.npy"

    board_src = project_root / config["paths"]["board_file"]
    shutil.copy2(board_src, board_out)

    min_corners = 12
    num_cams = 7

    print_section("Pipeline2 full-sensor Lensboy -> Calibcam detection converter MIN12")
    print(f"Lensboy detection dir: {lensboy_det_dir}")
    print(f"Output dir:            {output_dir}")
    print(f"Min corners per frame: {min_corners}")
    print("Offset correction:     NONE")

    detections_per_cam = []

    for cam_index in range(num_cams):
        npz_path = lensboy_det_dir / f"cam{cam_index}_lensboy_fullsensor_detections.npz"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing Lensboy detection file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        frame_numbers = data["detected_frame_numbers"].astype(int)
        corner_ids_list = data["corner_ids_list"]
        corner_xy_list = data["corner_xy_list"]

        marker_coords = []
        marker_ids = []
        detection_idxs = []
        frame_idxs = []

        for det_i, frame_idx in enumerate(frame_numbers):
            ids = np.asarray(corner_ids_list[det_i], dtype=np.int32).reshape(-1)
            xy = np.asarray(corner_xy_list[det_i], dtype=np.float64).reshape(-1, 2)

            if len(ids) != len(xy):
                raise ValueError(
                    f"cam{cam_index} frame {frame_idx}: ids/xy length mismatch "
                    f"{len(ids)} vs {len(xy)}"
                )

            if len(ids) < min_corners:
                continue

            marker_ids.append(ids.tolist())
            marker_coords.append(xy.reshape(-1, 1, 2))

            frame_idxs.append(int(frame_idx))
            detection_idxs.append(int(frame_idx))

        cam_detection_dict = {
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        }

        det_cam = Detections.from_list(cam_detection_dict)
        detections_per_cam.append(det_cam)

        print(
            f"cam{cam_index}: kept {len(frame_idxs)} frames "
            f"from {len(frame_numbers)} detected frames"
        )

    detections_all = sum(detections_per_cam, Detections())

    print_section("Combined detection object")
    print("Number of cameras:", detections_all.get_n_cams())
    print("Number of frames:", detections_all.get_n_frames())
    print("Number of markers:", detections_all.get_n_markers())
    print("Number of dimensions:", detections_all.get_n_dim())

    print("Saving detection.yml ...")
    detections_all.to_file(detections_out)

    print_section("Reload test")
    detection_files = sorted(output_dir.glob("detection_*.yml"))

    for p in detection_files:
        print(" -", p)

    loaded = Detections.from_file(detection_files)
    print("Reloaded number of cameras:", loaded.get_n_cams())
    print("Reloaded number of frames:", loaded.get_n_frames())
    print("Reloaded number of markers:", loaded.get_n_markers())

    print_section("Saved files")
    print(f"Detection file base: {detections_out}")
    print(f"Board copy:          {board_out}")
    print("Done.")


if __name__ == "__main__":
    main()
