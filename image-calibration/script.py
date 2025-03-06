import json
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

# Read the calibration parameters from a file
def read_calib_parameters(file_path):
    try:
        with open(file_path, 'r') as file_stream:
            json_struct = json.load(file_stream)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False, None, None, None, None

    assert json_struct["Calibration"]["cameras"][0]["model"]["polymorphic_name"] == "libCalib::CameraModelOpenCV"

    n_cameras = len(json_struct["Calibration"]["cameras"])

    if n_cameras < 1:
        return False, None, None, None, None

    K = [None] * n_cameras
    k = [None] * n_cameras
    cam_rvecs = [None] * n_cameras
    cam_tvecs = [None] * n_cameras

    for i in range(n_cameras):
        intrinsics = json_struct["Calibration"]["cameras"][i]["model"]["ptr_wrapper"]["data"]["parameters"]

        f = intrinsics["f"]["val"]
        ar = intrinsics["ar"]["val"]
        cx = intrinsics["cx"]["val"]
        cy = intrinsics["cy"]["val"]
        k1 = intrinsics["k1"]["val"]
        k2 = intrinsics["k2"]["val"]
        k3 = intrinsics["k3"]["val"]
        k4 = intrinsics["k4"]["val"]
        k5 = intrinsics["k5"]["val"]
        k6 = intrinsics["k6"]["val"]
        p1 = intrinsics["p1"]["val"]
        p2 = intrinsics["p2"]["val"]
        s1 = intrinsics["s1"]["val"]
        s2 = intrinsics["s2"]["val"]
        s3 = intrinsics["s3"]["val"]
        s4 = intrinsics["s4"]["val"]
        tauX = intrinsics["tauX"]["val"]
        tauY = intrinsics["tauY"]["val"]

        K[i] = np.array([[f, 0.0, cx],
                         [0.0, f * ar, cy],
                         [0.0, 0.0, 1.0]])

        k[i] = np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY])

        transform = json_struct["Calibration"]["cameras"][i]["transform"]

        rot = transform["rotation"]
        cam_rvecs[i] = np.array([rot["rx"], rot["ry"], rot["rz"]])

        t = transform["translation"]
        cam_tvecs[i] = np.array([t["x"], t["y"], t["z"]])

    return True, K, k, cam_rvecs, cam_tvecs

def undistort_image(image, K, dist_coeffs):
    """
    Undistort an image using the camera matrix and distortion coefficients.

    :param image: Input distorted image.
    :param K: Camera matrix (3x3 numpy array).
    :param dist_coeffs: Distortion coefficients (1xN numpy array, where N is the number of coefficients).
    :return: Undistorted image.
    """
    h, w = image.shape[:2]

    # Get the optimal new camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_camera_matrix)

    # Crop the image to the region of interest (ROI) if needed
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image

def main():

    CALIBRATION_DATA_FILENAME = "calibration-params.json" # full name including .json extension
    DISTORTED_IMG_DIR = "distorted_images"
    UNDISTERTED_IMG_DIR = "undistorted_images"

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify a file path relative to the script directory
    calibration_params_file_path = os.path.join(script_dir, CALIBRATION_DATA_FILENAME)
    dist_image_dir_path = os.path.join(script_dir, DISTORTED_IMG_DIR)
    undist_image_dir_path = os.path.join(script_dir, UNDISTERTED_IMG_DIR)

    print("Calibration params file path:", calibration_params_file_path)
    print("Disterted Image directory path:", dist_image_dir_path)
    print("Undisterted Image directory path:", undist_image_dir_path)

    # Convert dist_image_dir_path into Path object
    dist_image_dir_path = Path(dist_image_dir_path)
    undist_image_dir_path = Path(undist_image_dir_path)

    # Read the calibration parameters
    success, K, k, cam_rvecs, cam_tvecs = read_calib_parameters(calibration_params_file_path)

    if not success:
        print("Failed to read calibration parameters.")
        return

    # Iterate through all images in the input directory
    for image_path in tqdm(list(dist_image_dir_path.glob("*.*"))):  # Match all files
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:  # Check for image files
            print(f"Processing {image_path.name}...")

            # Load the distorted image
            distorted_image = cv2.imread(str(image_path))
            if distorted_image is None:
                print(f"Error: Unable to load image at {image_path}")
                continue

            # Extract the distortion coefficients (k1, k2, p1, p2, k3, ...)
            dist_coeffs = k[0][:8]  # Use the first camera's distortion coefficients

            # Undistort the image
            undistorted_image = undistort_image(distorted_image, K[0], dist_coeffs)

            # Save the undistorted image to the output directory
            output_image_path = undist_image_dir_path / ("undistorted_" + image_path.name)
            cv2.imwrite(str(output_image_path), undistorted_image)
            print(f"Saved undistorted image to {output_image_path}")

    print("All images processed.")

if __name__ == "__main__":
    main()