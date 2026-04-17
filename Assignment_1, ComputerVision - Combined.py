import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

### Paths ###
image_folder=r"C:\Users\noahg\Downloads\Oxford-IIIT Pet Subset\Oxford-IIIT Pet Subset\image"
annotation_folder=r"C:\Users\noahg\Downloads\Oxford-IIIT Pet Subset\Oxford-IIIT Pet Subset\annotation"

### Load files ###
image_files=[f for f in os.listdir(image_folder) if f.endswith((".jpg",".jpeg",".png"))]
print(f"Total images found: {len(image_files)}")

### Accumulators ###
total_iou=0
total_dice=0
count=0

### Main loop ###
for i,file in enumerate(image_files):
    print(f"\nProcessing: {file}")

    # Load image
    image_path=os.path.join(image_folder,file)
    image=cv2.imread(image_path)

    # Load mask
    name=os.path.splitext(file)[0]
    mask_path=os.path.join(annotation_folder,name+".png")
    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    binary_mask=(mask==1).astype("uint8")

    # Convert to grayscale
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Smooth the image
    blur=cv2.GaussianBlur(gray,(5,5),0)

    # Detect edges
    edges=cv2.Canny(blur,30,100)

    # Clean up edges
    kernel=np.ones((3,3),np.uint8)
    edges=cv2.dilate(edges,kernel,iterations=1)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel,iterations=2)

    # Find outer contours and keep the largest one
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour_mask=np.zeros_like(gray)
    if len(contours)>0:
        largest_contour=max(contours,key=cv2.contourArea)
        cv2.drawContours(contour_mask,[largest_contour],-1,255,thickness=cv2.FILLED)

    # Create foreground and background regions
    sure_fg=cv2.erode(contour_mask,kernel,iterations=2)
    sure_bg=cv2.dilate(contour_mask,kernel,iterations=3)

    # Identify unknown region
    unknown=cv2.subtract(sure_bg,sure_fg)

    # Create markers for watershed
    _,markers=cv2.connectedComponents(sure_fg,connectivity=8)
    markers=markers+1
    markers[unknown==255]=0

    # Apply watershed
    watershed_image=image.copy()
    cv2.watershed(watershed_image,markers)

    segmentation_mask=np.zeros_like(gray)
    segmentation_mask[markers>1]=255
    segmentation_mask[markers==-1]=0

    # Evaluate performance
    pred=(segmentation_mask>0).astype(np.uint8)
    gt=(binary_mask>0).astype(np.uint8)

    intersection=np.logical_and(pred,gt).sum()
    union=np.logical_or(pred,gt).sum()
    iou=intersection/union if union!=0 else 0
    dice=(2*intersection)/(pred.sum()+gt.sum()) if (pred.sum()+gt.sum())!=0 else 0

    print(f"IoU: {iou:.4f}, Dice: {dice:.4f}")

    total_iou+=iou
    total_dice+=dice
    count+=1

### Final results ###
if count>0:
    print("\nGlobal results")
    print(f"Average IoU: {total_iou/count:.4f}")
    print(f"Average Dice: {total_dice/count:.4f}")