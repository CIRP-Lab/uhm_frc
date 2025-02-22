import os
import re
import cv2
import multiprocessing as mp
from imagealigner import align_images
# Function to extract the unique frame identifier from filenames
def extract_frame_id(filename):
    match = re.match(r"(DJI_\d{14}_\d{4})", filename)
    return match.group(1) if match else None



def process_pair(ref_path, target_path):
    ref_img = cv2.imread(ref_path)
    target_img = cv2.imread(target_path)
    
    if ref_img is None or target_img is None:
        print(f"Skipping {target_path}, could not read image.")
        return
    
    aligned_img, H = align_images(ref_img, target_img)  # Unpack both returned values

    # Save the aligned image
    output_filename = os.path.basename(target_path).replace(".TIF", "_aligned.jpg")
    output_path = os.path.join("aligned_output", output_filename)
    cv2.imwrite(output_path, aligned_img)
    
    # Save the homography matrix (H) in a text file
    H_filename = output_filename.replace(".jpg", "_homography.txt")
    H_path = os.path.join("aligned_output", H_filename)
    
    with open(H_path, "w") as f:
        for row in H:
            f.write(" ".join(map(str, row)) + "\n")

    print(f"Saved: {output_path} and {H_path}")


def main(ref_folder, non_ref_folder, output_folder="aligned_output"):
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all reference and non-reference images
    ref_images = {extract_frame_id(f): os.path.join(ref_folder, f) 
                  for f in os.listdir(ref_folder) if f.endswith("_D.JPG")}
    
    non_ref_images = [(extract_frame_id(f), os.path.join(non_ref_folder, f)) 
                      for f in os.listdir(non_ref_folder) if f.endswith(".TIF")]

    # Create processing pairs (only for matching frame IDs)
    image_pairs = [(ref_images[frame_id], img_path) for frame_id, img_path in non_ref_images if frame_id in ref_images]
    
    print(f"Found {len(image_pairs)} matching image pairs.")

    # Process in parallel
    i=0
    for ref_path, target_path in image_pairs:
        process_pair(ref_path, target_path)
        if (i > 3): # so that it stops after aligning a few for testing
            break
        i+=1

if __name__ == "__main__":
    main("imagealigner/images/rgb", "imagealigner/images/notrgb")
