import cv2
import numpy as np

def align_images(base_image, target_image, max_matches=50):
    # Convert images to grayscale
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features and compute descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(base_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target_gray, None)

    # Match features using FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe’s ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Limit the number of matches to max_matches
    #good_matches = good_matches[:max_matches]

    if len(good_matches) > 4:
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp target image to align with base image
        aligned_image = cv2.warpPerspective(target_image, H, (base_image.shape[1], base_image.shape[0]))

        # Draw the limited number of matches between the keypoints
        output_image = cv2.drawMatches(base_image, keypoints1, target_image, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Resize the output image for easier viewing
        output_image_resized = cv2.resize(output_image, (800, 600))  # Adjust the size as needed

        # Show the intermediate result (matches)
        cv2.imshow("Matches", output_image_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return aligned_image, H
    else:
        print("Not enough good matches found!")
        return target_image, None

if __name__ == "__main__":
    base_img = cv2.imread("CamRE211.TIF")
    target_img = cv2.imread("CAMG211.TIF")

    aligned_img, H = align_images(base_img, target_img)

    if H is not None:
        cv2.imwrite("aligned_camera_B.jpg", aligned_img)
        print("Image successfully aligned and saved!")
