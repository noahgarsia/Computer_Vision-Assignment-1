import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

### Paths ###
image_folder = r"C:\Users\noahg\Downloads\Oxford-IIIT Pet Subset\Oxford-IIIT Pet Subset\image"
annotation_folder = r"C:\Users\noahg\Downloads\Oxford-IIIT Pet Subset\Oxford-IIIT Pet Subset\annotation"

### Load files ###
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
print(f"Total images found: {len(image_files)}")

### First image only ###
if len(image_files) > 0:
    first_file = image_files[0]
    first_image_path = os.path.join(image_folder, first_file)
    first_image = cv2.imread(first_image_path)

    if first_image is not None:
        first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        first_blur = cv2.GaussianBlur(first_gray, (5, 5), 0)

        # Show gray image
        plt.figure(figsize=(6, 6))
        plt.imshow(first_gray, cmap="gray")
        plt.title(f"First Image Gray - {first_file}")
        plt.axis("off")
        plt.show()

        # Manual thresholds
        threshold_values = [40, 60, 80, 100, 120, 140]
        plt.figure(figsize=(18, 10))
        plt.subplot(2, 4, 1)
        plt.imshow(first_gray, cmap="gray")
        plt.title("Gray")
        plt.axis("off")

        for idx, t in enumerate(threshold_values, start=2):
            _, manual_thresh = cv2.threshold(first_blur, t, 255, cv2.THRESH_BINARY_INV)
            plt.subplot(2, 4, idx)
            plt.imshow(manual_thresh, cmap="gray")
            plt.title(f"Threshold = {t}")
            plt.axis("off")

        # Otsu
        _, otsu_thresh_display = cv2.threshold(first_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        plt.subplot(2, 4, 8)
        plt.imshow(otsu_thresh_display, cmap="gray")
        plt.title("Otsu")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

### Accumulators ###
total_iou = 0
total_dice = 0
count = 0

results = []   

### Main loop ###
for i, file in enumerate(image_files):
    print(f"\nProcessing: {file}")

    # Load image
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask
    name = os.path.splitext(file)[0]
    mask_path = os.path.join(annotation_folder, name + ".png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask == 1).astype("uint8")

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Foreground and background
    sure_fg = cv2.erode(thresh, kernel, iterations=2)
    sure_bg = cv2.dilate(thresh, kernel, iterations=2)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    cv2.watershed(image, markers)
    segmentation_mask = np.zeros_like(gray)
    segmentation_mask[markers > 1] = 255
    segmentation_mask[markers == -1] = 0

    # Display
    if i < 5:
        plt.imshow(segmentation_mask, cmap="gray")
        plt.title("Watershed (Automatic Markers)")
        plt.axis("off")
        plt.show()

    # Evaluation
    pred = (segmentation_mask > 0).astype(np.uint8)
    gt = (binary_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else 0

    print(f"IoU: {iou:.4f}, Dice: {dice:.4f}")

    total_iou += iou
    total_dice += dice
    count += 1

    results.append((file, iou, dice))   

### Final ###
if count > 0:
    print("\nFinal results")
    print(f"Average IoU: {total_iou / count:.4f}")
    print(f"Average Dice: {total_dice / count:.4f}")

    # Table of results
    print("\nResults table")
    print(f"{'Image':<30} {'IoU':<10} {'Dice':<10}")
    print("-" * 50)

    for file, iou, dice in results[:10]:
        print(f"{file:<30} {iou:<10.4f} {dice:<10.4f}")