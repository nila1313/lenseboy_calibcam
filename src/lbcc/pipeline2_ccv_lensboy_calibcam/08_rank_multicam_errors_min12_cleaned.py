from pathlib import Path
import csv
import yaml
import numpy as np
import cv2


PROJECT_ROOT = Path("/Users/nilamaitrachaity/Desktop/lenseboy_calibcam")

CALIB_PATH = PROJECT_ROOT / "runs/pipeline2/11_calibcam_all7_pinhole_from_fullsensor_lensboy_min12_cleaned/multicam_calibration.npy"
BOARD_POSES_PATH = PROJECT_ROOT / "runs/pipeline2/11_calibcam_all7_pinhole_from_fullsensor_lensboy_min12_cleaned/multicam_calibration_board_positions.yml"
DET_DIR = PROJECT_ROOT / "runs/pipeline2/10_calibcam_from_fullsensor_lensboy_min12_cleaned"

OUT_DIR = PROJECT_ROOT / "runs/pipeline2/13_multicam_reprojection_diagnostics_min12_cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_SIZE_REAL = 3.0
INNER_COLS = 5
INNER_ROWS = 7
N_MARKERS = 35


def make_object_points():
    pts = []
    for marker_id in range(N_MARKERS):
        col = marker_id % INNER_COLS
        row = marker_id // INNER_COLS
        x = (col + 1) * SQUARE_SIZE_REAL
        y = (row + 1) * SQUARE_SIZE_REAL
        pts.append([x, y, 0.0])
    return np.asarray(pts, dtype=float)


def R_from_rvec(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
    return R


def project_points_manual(X_cam, A, k):
    A = np.asarray(A, dtype=float)
    k = np.asarray(k, dtype=float).reshape(-1)

    fx = A[0, 0]
    fy = A[1, 1]
    cx = A[0, 2]
    cy = A[1, 2]

    k1 = k[0] if len(k) > 0 else 0.0
    k2 = k[1] if len(k) > 1 else 0.0
    p1 = k[2] if len(k) > 2 else 0.0
    p2 = k[3] if len(k) > 3 else 0.0
    k3 = k[4] if len(k) > 4 else 0.0

    z = X_cam[:, 2]
    x = X_cam[:, 0] / z
    y = X_cam[:, 1] / z

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = fx * x_dist + cx
    v = fy * y_dist + cy

    return np.stack([u, v], axis=1)


def load_detection(cam_idx):
    p = DET_DIR / f"detection_{cam_idx:03d}.yml"
    with open(p, "r") as f:
        det = yaml.safe_load(f)

    frame_idxs = np.asarray(det["frame_idxs"], dtype=int).reshape(-1)
    marker_ids = np.asarray(det["marker_ids"], dtype=int).reshape(-1)
    marker_coords = np.asarray(det["marker_coords"], dtype=float)

    if marker_coords.ndim == 4:
        marker_coords = marker_coords[0]

    frame_to_slot = {}
    for slot_idx, frame_idx in enumerate(frame_idxs):
        if frame_idx >= 0:
            frame_to_slot[int(frame_idx)] = int(slot_idx)

    return frame_to_slot, marker_ids, marker_coords


def transform_points(mode, X_board, R_board, t_board, R_cam, t_cam):
    """
    We test a few possible pose conventions.
    The correct one should reproduce Calibcam's final max/median table.
    """

    # Board local -> world/cam0
    X_world = X_board @ R_board.T + t_board.reshape(1, 3)

    if mode == "calibcam_row":
        # Matches calibcamlib/camerasystem row-vector style:
        # X_cam = X_world @ R_cam.T + t_cam
        X_cam = X_world @ R_cam.T + t_cam.reshape(1, 3)

    elif mode == "inverse_cam":
        # Alternative camera-pose convention:
        # X_cam = (X_world - t_cam) @ R_cam
        X_cam = (X_world - t_cam.reshape(1, 3)) @ R_cam

    elif mode == "cam_R_no_inverse":
        # Another possible convention.
        X_cam = X_world @ R_cam + t_cam.reshape(1, 3)

    elif mode == "board_only":
        # Useful sanity check. Only correct for cam0.
        X_cam = X_world

    else:
        raise ValueError(mode)

    return X_cam


def evaluate_mode(mode, calib, board_data, detections, object_points):
    calibs = calib["calibs"]

    board_frame_idxs = np.asarray(board_data["frame_idxs"], dtype=int)
    board_rvecs = np.asarray(board_data["rvecs"], dtype=float)
    board_tvecs = np.asarray(board_data["tvecs"], dtype=float)

    frame_rows = []
    point_rows = []

    for pose_idx in range(board_rvecs.shape[0]):
        R_board = R_from_rvec(board_rvecs[pose_idx])
        t_board = board_tvecs[pose_idx]

        for cam_idx in range(7):
            frame_idx = int(board_frame_idxs[cam_idx, pose_idx])
            if frame_idx < 0:
                continue

            frame_to_slot, marker_ids, marker_coords = detections[cam_idx]
            if frame_idx not in frame_to_slot:
                continue

            det_slot = frame_to_slot[frame_idx]
            coords_all = marker_coords[det_slot]

            finite = np.isfinite(coords_all[:, 0]) & np.isfinite(coords_all[:, 1])
            if np.sum(finite) < 4:
                continue

            ids = marker_ids[finite]
            observed = coords_all[finite]
            X_board = object_points[ids]

            cam = calibs[cam_idx]
            A = np.asarray(cam["A"], dtype=float)
            k = np.asarray(cam["k"], dtype=float)
            R_cam = R_from_rvec(cam["rvec_cam"])
            t_cam = np.asarray(cam["tvec_cam"], dtype=float)

            X_cam = transform_points(mode, X_board, R_board, t_board, R_cam, t_cam)

            valid_z = np.isfinite(X_cam[:, 2]) & (np.abs(X_cam[:, 2]) > 1e-9)
            if not np.any(valid_z):
                continue

            X_cam = X_cam[valid_z]
            observed_valid = observed[valid_z]
            ids_valid = ids[valid_z]

            try:
                projected = project_points_manual(X_cam, A, k)
            except Exception:
                continue

            good = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
            if not np.any(good):
                continue

            projected = projected[good]
            observed_valid = observed_valid[good]
            ids_valid = ids_valid[good]

            errors = np.linalg.norm(projected - observed_valid, axis=1)

            frame_rows.append({
                "mode": mode,
                "camera_index": cam_idx,
                "pose_index": int(pose_idx),
                "frame_idx": int(frame_idx),
                "n_points": int(len(errors)),
                "mean_error_px": float(np.mean(errors)),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(np.max(errors)),
                "num_points_gt_5px": int(np.sum(errors > 5.0)),
                "num_points_gt_10px": int(np.sum(errors > 10.0)),
                "num_points_gt_20px": int(np.sum(errors > 20.0)),
            })

            for i, marker_id in enumerate(ids_valid):
                point_rows.append({
                    "mode": mode,
                    "camera_index": cam_idx,
                    "pose_index": int(pose_idx),
                    "frame_idx": int(frame_idx),
                    "marker_id": int(marker_id),
                    "observed_x": float(observed_valid[i, 0]),
                    "observed_y": float(observed_valid[i, 1]),
                    "projected_x": float(projected[i, 0]),
                    "projected_y": float(projected[i, 1]),
                    "error_px": float(errors[i]),
                })

    return frame_rows, point_rows


def summarize_mode(mode, frame_rows, point_rows):
    print("")
    print("=" * 80)
    print("MODE:", mode)
    print("=" * 80)

    if not point_rows:
        print("No valid points.")
        return float("inf")

    all_errors = np.asarray([r["error_px"] for r in point_rows], dtype=float)
    print("global median:", float(np.median(all_errors)))
    print("global max:", float(np.max(all_errors)))
    print("n points:", len(point_rows))

    print("")
    print("cam | n_points | max | median")
    score = 0.0

    for cam_idx in range(7):
        cam_errors = np.asarray(
            [r["error_px"] for r in point_rows if r["camera_index"] == cam_idx],
            dtype=float,
        )

        if cam_errors.size == 0:
            print(f"{cam_idx:3d} | {0:8d} | {'NA':>8s} | {'NA':>8s}")
            score += 1e9
            continue

        cam_max = float(np.max(cam_errors))
        cam_med = float(np.median(cam_errors))
        score += cam_med

        print(f"{cam_idx:3d} | {cam_errors.size:8d} | {cam_max:8.2f} | {cam_med:8.2f}")

    return score


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 80)
    print("True multi-camera reprojection diagnostics using board poses")
    print("=" * 80)

    calib = np.load(CALIB_PATH, allow_pickle=True).item()

    with open(BOARD_POSES_PATH, "r") as f:
        board_data = yaml.safe_load(f)

    object_points = make_object_points()
    detections = [load_detection(cam_idx) for cam_idx in range(7)]

    print("Calibration:", CALIB_PATH)
    print("Board poses:", BOARD_POSES_PATH)
    print("Detection dir:", DET_DIR)
    print("Output dir:", OUT_DIR)
    print("Board frame_idxs shape:", np.asarray(board_data["frame_idxs"]).shape)
    print("Board rvecs shape:", np.asarray(board_data["rvecs"]).shape)
    print("Board tvecs shape:", np.asarray(board_data["tvecs"]).shape)

    modes = ["calibcam_row", "inverse_cam", "cam_R_no_inverse", "board_only"]

    results = []

    for mode in modes:
        frame_rows, point_rows = evaluate_mode(mode, calib, board_data, detections, object_points)
        score = summarize_mode(mode, frame_rows, point_rows)
        results.append((score, mode, frame_rows, point_rows))

    results.sort(key=lambda x: x[0])
    best_score, best_mode, best_frame_rows, best_point_rows = results[0]

    print("")
    print("=" * 80)
    print("SELECTED BEST MODE:", best_mode)
    print("=" * 80)

    top_frames = sorted(best_frame_rows, key=lambda r: r["max_error_px"], reverse=True)[:100]
    top_points = sorted(best_point_rows, key=lambda r: r["error_px"], reverse=True)[:300]

    print("")
    print("TOP 40 BAD MULTICAM FRAMES")
    print("cam | frame | pose | n_pts | median | max | >10px | >20px")
    for r in top_frames[:40]:
        print(
            f"{r['camera_index']:3d} | "
            f"{r['frame_idx']:5d} | "
            f"{r['pose_index']:4d} | "
            f"{r['n_points']:5d} | "
            f"{r['median_error_px']:6.2f} | "
            f"{r['max_error_px']:6.2f} | "
            f"{r['num_points_gt_10px']:5d} | "
            f"{r['num_points_gt_20px']:5d}"
        )

    print("")
    print("TOP 40 BAD MULTICAM POINTS")
    print("cam | frame | pose | marker | error")
    for r in top_points[:40]:
        print(
            f"{r['camera_index']:3d} | "
            f"{r['frame_idx']:5d} | "
            f"{r['pose_index']:4d} | "
            f"{r['marker_id']:6d} | "
            f"{r['error_px']:8.2f}"
        )

    write_csv(OUT_DIR / "per_frame_multicam_errors.csv", best_frame_rows)
    write_csv(OUT_DIR / "per_point_multicam_errors.csv", best_point_rows)
    write_csv(OUT_DIR / "top_bad_multicam_frames.csv", top_frames)
    write_csv(OUT_DIR / "top_bad_multicam_points.csv", top_points)

    print("")
    print("Saved:")
    print(" ", OUT_DIR / "per_frame_multicam_errors.csv")
    print(" ", OUT_DIR / "per_point_multicam_errors.csv")
    print(" ", OUT_DIR / "top_bad_multicam_frames.csv")
    print(" ", OUT_DIR / "top_bad_multicam_points.csv")


if __name__ == "__main__":
    main()
