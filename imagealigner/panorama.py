import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# def stitch_images(images):
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()

#     # Step 1: Detect keypoints and descriptors for each image
#     keypoints_list = []
#     descriptors_list = []
#     for img in images:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         keypoints, descriptors = sift.detectAndCompute(gray, None)
#         keypoints_list.append(keypoints)
#         descriptors_list.append(descriptors)

#     # Step 2: Find matches between consecutive images
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches_list = []
#     for i in range(len(images) - 1):
#         # Match descriptors between image i and image i+1
#         matches = bf.match(descriptors_list[i], descriptors_list[i + 1])
#         matches = sorted(matches, key=lambda x: x.distance)
#         matches_list.append(matches)

#     # Step 3: Compute homography matrices for stitching
#     homographies = [np.eye(3)]  # Initialize first image with identity matrix
#     for i in range(len(matches_list)):
#         src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
#         dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)

#         # Find the homography matrix
#         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         homographies.append(H)

#     # Step 4: Warp images to align them based on homographies
#     result = images[0]
#     for i in range(1, len(images)):
#         # Warp each subsequent image into the coordinate system of the first one
#         result = cv2.warpPerspective(result, homographies[i], (result.shape[1] + images[i].shape[1], result.shape[0]))
#         result[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

#     return result

        

def resize_images(images, max_dim=3000):
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_images.append(cv2.resize(img, (new_w, new_h)))
        else:
            resized_images.append(img)

    return resized_images

def stitch_images(images):
    # cv2.ocl.setUseOpenCL(False)

    stitcher = cv2.Stitcher.create()
    stitcher.setWaveCorrection(False)
    (status, stitched_image) = stitcher.stitch(images)
    if status == cv2.STITCHER_OK:
        return stitched_image
    else:
        return None

    

folder_path = 'imagealigner/pano'  # Replace with your folder path
images = []
for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path,filename))
    if img is not None:
        images.append(img)
# images = resize_images(images)
panorama = stitch_images(images)

# Show the resulting panorama
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Optionally, save the panorama
cv2.imwrite('panorama_output.jpg', panorama)
