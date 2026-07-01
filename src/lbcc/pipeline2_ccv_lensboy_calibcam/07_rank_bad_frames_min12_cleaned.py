from pathlib import Path
import csv
import yaml
import numpy as np
import cv2


PROJECT_ROOT = Path("/Users/nilamaitrachaity/Desktop/lenseboy_calibcam")

CALIB_PATH = PROJECT_ROOT / "runs/pipeline2/11_calibcam_all7_pinhole_from_fullsensor_lensboy_min12_cleaned/multicam_calibration.npy"
DET_DIR = PROJECT_ROOT / "runs/pipeline2/10_calibcam_from_fullsensor_lensboy_min12_cleaned"
OUT_DIR = PROJECT_ROOT / "runs/pipeline2/12_reprojection_diagnostics_min12_cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_SIZE_REAL = 3.0
INNER_COLS = 5
INNER_ROWS = 7
N_MARKERS = 35


def make_object_points(layout_name):
    pts = []

    for marker_id in range(N_MARKERS):
        col = marker_id % INNER_COLS
        row = marker_id // INNER_COLS

        x = (col + 1) * SQUARE_SIZE_REAL
        y = (row + 1) * SQUARE_SIZE_REAL

        if "flipx" in layout_name:
            x = (INNER_COLS - col) * SQUARE_SIZE_REAL

        if "flipy" in layout_name:
            y = (INNER_ROWS - row) * SQUARE_SIZE_REAL

        pts.append([x, y, 0.0])

    return np.asarray(pts, dtype=np.float32)


def pnp_errors(obj_pts, img_pts, A, dist):
    if len(obj_pts) < 6:
        return None

    obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        A,
        dist,
        iterationsCount=100,
        reprojectionError=8.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            A,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None

    if not ok:
        return None

    # Refine using all points after initial pose.
    try:
        ok2, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            A,
            dist,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        pass

    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, A, dist)
    projected = projected.reshape(-1, 2)

    err = np.linalg.norm(projected - img_pts, axis=1)

    return {
        "rvec": rvec,
        "tvec": tvec,
        "errors": err,
        "projected": projected,
        "inliers": inliers,
    }


def load_detection(cam_idx):
    p = DET_DIR / f"detection_{cam_idx:03d}.yml"

    with open(p, "r") as f:
        det = yaml.safe_load(f)

    frame_idxs = np.asarray(det["frame_idxs"], dtype=int).reshape(-1)
    marker_ids = np.asarray(det["marker_ids"], dtype=int).reshape(-1)
    marker_coords = np.asarray(det["marker_coords"], dtype=float)

    # Shape is usually (1, 354, 35, 2). Make it (354, 35, 2).
    if marker_coords.ndim == 4:
        marker_coords = marker_coords[0]

    return p, frame_idxs, marker_ids, marker_coords


def main():
    print("=" * 80)
    print("Pipeline2 min12 reprojection diagnostics")
    print("=" * 80)

    calib = np.load(CALIB_PATH, allow_pickle=True).item()
    calibs = calib["calibs"]

    print("Calibration:", CALIB_PATH)
    print("Detection dir:", DET_DIR)
    print("Output dir:", OUT_DIR)
    print("Number of camera calibs:", len(calibs))

    layout_names = ["normal", "flipx", "flipy", "flipx_flipy"]
    layout_points = {name: make_object_points(name) for name in layout_names}

    # First: choose the layout with lowest median PnP reprojection error.
    layout_scores = []

    for layout_name, object_points_all in layout_points.items():
        all_medians = []

        for cam_idx in range(7):
            cam = calibs[cam_idx]
            A = np.asarray(cam["A"], dtype=np.float64)
            dist = np.asarray(cam["k"], dtype=np.float64).reshape(-1, 1)

            _, frame_idxs, marker_ids, marker_coords = load_detection(cam_idx)

            for slot_idx, frame_idx in enumerate(frame_idxs):
                if frame_idx < 0:
                    continue

                coords_all = marker_coords[slot_idx]
                finite = np.isfinite(coords_all[:, 0]) & np.isfinite(coords_all[:, 1])

                ids = marker_ids[finite]
                img_pts = coords_all[finite]
                obj_pts = object_points_all[ids]

                result = pnp_errors(obj_pts, img_pts, A, dist)
                if result is None:
                    continue

                all_medians.append(float(np.median(result["errors"])))

        score = float(np.median(all_medians)) if all_medians else float("inf")
        layout_scores.append((layout_name, score))

    layout_scores = sorted(layout_scores, key=lambda x: x[1])
    best_layout_name = layout_scores[0][0]
    object_points_all = layout_points[best_layout_name]

    print("")
    print("Layout test:")
    for name, score in layout_scores:
        print(f"  {name:12s}: median PnP error = {score:.4f} px")

    print("")
    print("Selected layout:", best_layout_name)

    frame_rows = []
    point_rows = []

    for cam_idx in range(7):
        cam = calibs[cam_idx]
        A = np.asarray(cam["A"], dtype=np.float64)
        dist = np.asarray(cam["k"], dtype=np.float64).reshape(-1, 1)

        det_path, frame_idxs, marker_ids, marker_coords = load_detection(cam_idx)

        print("")
        print("=" * 80)
        print(f"cam{cam_idx}")
        print("Detection:", det_path)

        for slot_idx, frame_idx in enumerate(frame_idxs):
            if frame_idx < 0:
                continue

            coords_all = marker_coords[slot_idx]
            finite = np.isfinite(coords_all[:, 0]) & np.isfinite(coords_all[:, 1])

            ids = marker_ids[finite]
            img_pts = coords_all[finite]
            obj_pts = object_points_all[ids]

            result = pnp_errors(obj_pts, img_pts, A, dist)
            if result is None:
                continue

            errors = result["errors"]
            projected = result["projected"]
            inliers = result["inliers"]

            if inliers is None:
                n_inliers = -1
            else:
                n_inliers = int(len(inliers))

            frame_rows.append({
                "camera_index": cam_idx,
                "slot_index": int(slot_idx),
                "frame_idx": int(frame_idx),
                "n_points": int(len(errors)),
                "n_ransac_inliers": n_inliers,
                "mean_error_px": float(np.mean(errors)),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(np.max(errors)),
                "num_points_gt_5px": int(np.sum(errors > 5.0)),
                "num_points_gt_10px": int(np.sum(errors > 10.0)),
                "num_points_gt_20px": int(np.sum(errors > 20.0)),
            })

            for local_i, marker_id in enumerate(ids):
                point_rows.append({
                    "camera_index": cam_idx,
                    "slot_index": int(slot_idx),
                    "frame_idx": int(frame_idx),
                    "marker_id": int(marker_id),
                    "observed_x": float(img_pts[local_i, 0]),
                    "observed_y": float(img_pts[local_i, 1]),
                    "projected_x": float(projected[local_i, 0]),
                    "projected_y": float(projected[local_i, 1]),
                    "error_px": float(errors[local_i]),
                })

        cam_rows = [r for r in frame_rows if r["camera_index"] == cam_idx]
        if cam_rows:
            max_e = max(r["max_error_px"] for r in cam_rows)
            med_e = float(np.median([r["median_error_px"] for r in cam_rows]))
            bad_frames = sum(r["max_error_px"] > 20 for r in cam_rows)
            print(f"frames evaluated: {len(cam_rows)}")
            print(f"median of frame medians: {med_e:.3f} px")
            print(f"max frame error: {max_e:.3f} px")
            print(f"frames with max error > 20px: {bad_frames}")

    frame_csv = OUT_DIR / "per_frame_pnp_errors.csv"
    point_csv = OUT_DIR / "per_point_pnp_errors.csv"
    top_frame_csv = OUT_DIR / "top_bad_frames.csv"
    top_point_csv = OUT_DIR / "top_bad_points.csv"

    with open(frame_csv, "w", newline="") as f:
        fieldnames = [
            "camera_index",
            "slot_index",
            "frame_idx",
            "n_points",
            "n_ransac_inliers",
            "mean_error_px",
            "median_error_px",
            "max_error_px",
            "num_points_gt_5px",
            "num_points_gt_10px",
            "num_points_gt_20px",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    with open(point_csv, "w", newline="") as f:
        fieldnames = [
            "camera_index",
            "slot_index",
            "frame_idx",
            "marker_id",
            "observed_x",
            "observed_y",
            "projected_x",
            "projected_y",
            "error_px",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(point_rows)

    top_frames = sorted(frame_rows, key=lambda r: r["max_error_px"], reverse=True)[:100]
    with open(top_frame_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_frames[0].keys()))
        writer.writeheader()
        writer.writerows(top_frames)

    top_points = sorted(point_rows, key=lambda r: r["error_px"], reverse=True)[:200]
    with open(top_point_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_points[0].keys()))
        writer.writeheader()
        writer.writerows(top_points)

    print("")
    print("=" * 80)
    print("TOP 30 BAD FRAMES")
    print("=" * 80)
    print("cam | frame | n_pts | median | max | >10px | >20px")
    for r in top_frames[:30]:
        print(
            f"{r['camera_index']:3d} | "
            f"{r['frame_idx']:5d} | "
            f"{r['n_points']:5d} | "
            f"{r['median_error_px']:6.2f} | "
            f"{r['max_error_px']:6.2f} | "
            f"{r['num_points_gt_10px']:5d} | "
            f"{r['num_points_gt_20px']:5d}"
        )

    print("")
    print("=" * 80)
    print("TOP 30 BAD POINTS")
    print("=" * 80)
    print("cam | frame | marker | error")
    for r in top_points[:30]:
        print(
            f"{r['camera_index']:3d} | "
            f"{r['frame_idx']:5d} | "
            f"{r['marker_id']:6d} | "
            f"{r['error_px']:8.2f}"
        )

    print("")
    print("Saved:")
    print(" ", frame_csv)
    print(" ", point_csv)
    print(" ", top_frame_csv)
    print(" ", top_point_csv)


if __name__ == "__main__":
    main()
