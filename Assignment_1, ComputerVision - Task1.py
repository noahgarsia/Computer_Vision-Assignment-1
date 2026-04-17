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

### Accumulators ###
total_iou = 0
total_dice = 0
count = 0

iou_values = []
dice_values = []

### Main loop ###
for i, file in enumerate(image_files):

    print(f"\nProcessing: {file}")

    # Load image
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {file}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if i < 5:
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")
        plt.show()

    # Load annotation
    image_name = os.path.splitext(file)[0]
    annotation_path = os.path.join(annotation_folder, image_name + ".png")

    mask = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Failed to load annotation for {file}")
        continue

    binary_mask = (mask == 1).astype("uint8")

    if i < 5:
        plt.imshow(binary_mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.show()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if i < 5:
        plt.imshow(gray, cmap="gray")
        plt.title("Grayscale Image")
        plt.axis("off")
        plt.show()

    # Canny edge detection
    edges = cv2.Canny(gray, 30, 100)

    if i < 5:
        plt.imshow(edges, cmap="gray")
        plt.title("Canny Edges")
        plt.axis("off")
        plt.show()

    # Morphology
    kernel = np.ones((3, 3), np.uint8)

    morphed_edges = cv2.dilate(edges, kernel, iterations=1)
    morphed_edges = cv2.morphologyEx(morphed_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    if i < 5:
        plt.imshow(morphed_edges, cmap="gray")
        plt.title("Morphology Output")
        plt.axis("off")
        plt.show()

    # Contour extraction
    contours, _ = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    contour_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    if i < 5:
        plt.imshow(contour_rgb)
        plt.title("All Contours")
        plt.axis("off")
        plt.show()

    # Segmentation mask
    segmentation_mask = np.zeros(gray.shape, dtype="uint8")

    if len(contours) > 0:
        cv2.drawContours(segmentation_mask, contours, -1, 255, thickness=cv2.FILLED)

    if i < 5:
        plt.imshow(segmentation_mask, cmap="gray")
        plt.title("Segmentation Mask")
        plt.axis("off")
        plt.show()

    # Evaluation
    pred = (segmentation_mask > 0).astype(np.uint8)
    gt = (binary_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union != 0 else 0

    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else 0

    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")

    total_iou += iou
    total_dice += dice
    count += 1

    iou_values.append(iou)
    dice_values.append(dice)

### Final averages ###
if count > 0:
    avg_iou = total_iou / count
    avg_dice = total_dice / count

    print("\nFinal results")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")

    ### Histograms ###
    plt.figure()
    plt.hist(iou_values, bins=10, edgecolor="black")
    plt.title("Histogram of IoU Values")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.hist(dice_values, bins=10, edgecolor="black")
    plt.title("Histogram of Dice Values")
    plt.xlabel("Dice Score")
    plt.ylabel("Frequency")
    plt.show()

else:
    print("No images were successfully processed.")