from pathlib import Path
import inspect

import cv2
import numpy as np
import yaml
import lensboy as lb


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    config_path = Path("configs/pipeline1_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = Path(config["paths"]["project_root"])
    board_path = project_root / config["paths"]["board_file"]

    print_section("Lensboy detection function")
    print("Function:")
    print(lb.extract_frames_from_charuco)

    print("\nSignature:")
    print(inspect.signature(lb.extract_frames_from_charuco))

    print("\nDocstring:")
    print(lb.extract_frames_from_charuco.__doc__)

    print_section("OpenCV ArUco availability")
    print("OpenCV version:", cv2.__version__)
    print("Has cv2.aruco:", hasattr(cv2, "aruco"))
    print("Has CharucoBoard:", hasattr(cv2.aruco, "CharucoBoard"))
    print("Has CharucoDetector:", hasattr(cv2.aruco, "CharucoDetector"))

    print_section("Board file")
    print("Board path:", board_path)
    print("Exists:", board_path.exists())

    data = np.load(board_path, allow_pickle=True)

    print("Raw loaded type:", type(data))
    print("Raw shape:", getattr(data, "shape", None))
    print("Raw dtype:", getattr(data, "dtype", None))

    obj = data.item() if getattr(data, "shape", None) == () else data

    print("\nAfter .item() if scalar:")
    print("Object type:", type(obj))
    print("Object repr:")
    print(obj)

    if isinstance(obj, dict):
        print("\nDictionary keys:")
        for k, v in obj.items():
            print(f"  {k}: {v} ({type(v)})")

    print_section("Try to build OpenCV CharucoBoard")

    if not isinstance(obj, dict):
        print("❌ Board object is not a dict. We need to inspect manually.")
        return

    board_width = int(obj.get("boardWidth", obj.get("board_width")))
    board_height = int(obj.get("boardHeight", obj.get("board_height")))

    square_length = float(obj.get("square_size_real", obj.get("squareLength", obj.get("square_size", 1.0))))
    marker_length = float(obj.get("marker_size", obj.get("markerLength", 0.6)))

    # If marker_size is stored as ratio and square_size_real exists, convert ratio to real unit.
    if "square_size_real" in obj and "marker_size" in obj:
        marker_length = float(obj["square_size_real"]) * float(obj["marker_size"])
        square_length = float(obj["square_size_real"])

    dictionary_type = int(obj.get("dictionary_type", obj.get("dictionary", cv2.aruco.DICT_4X4_50)))

    print("board_width:", board_width)
    print("board_height:", board_height)
    print("square_length:", square_length)
    print("marker_length:", marker_length)
    print("dictionary_type:", dictionary_type)

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)

    board = cv2.aruco.CharucoBoard(
        (board_width, board_height),
        square_length,
        marker_length,
        dictionary,
    )

    print("✅ OpenCV CharucoBoard created successfully")
    print("Chessboard corners shape:", board.getChessboardCorners().shape)
    print("Number of ChArUco corners:", len(board.getChessboardCorners()))


if __name__ == "__main__":
    main()
