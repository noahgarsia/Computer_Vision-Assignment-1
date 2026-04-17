import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# PATHS
image_folder=r"C:\Users\noahg\Downloads\Oxford-IIIT Pet Subset\Oxford-IIIT Pet Subset\image"
annotation_folder=r"C:\Users\noahg\Downloads\Oxford-IIIT Pet Subset\Oxford-IIIT Pet Subset\annotation"

# LOAD FILES
image_files=[f for f in os.listdir(image_folder) if f.endswith((".jpg",".jpeg",".png"))]
print(f"Total images found: {len(image_files)}")

# ACCUMULATORS
total_iou=0
total_dice=0

total_iou_rgb=0
total_dice_rgb=0

count=0

# MAIN LOOP
for i,file in enumerate(image_files):
    print(f"\nProcessing: {file}")

    # LOAD IMAGE
    image_path=os.path.join(image_folder,file)
    image=cv2.imread(image_path)
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # LOAD MASK
    name=os.path.splitext(file)[0]
    mask_path=os.path.join(annotation_folder,name+".png")
    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    binary_mask=(mask!=2).astype("uint8")

    # IMAGE SIZE
    h,w,_=image_rgb.shape

    # RGB ONLY
    pixel_values_rgb=image_rgb.reshape((-1,3))
    pixel_values_rgb=np.float32(pixel_values_rgb)

    k=4
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
    _,labels_rgb,centers_rgb=cv2.kmeans(pixel_values_rgb,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    labels_2d_rgb=labels_rgb.reshape((h,w))

    # CLUSTER SCORING RGB
    cluster_scores_rgb=[]
    image_center_x=w/2
    image_center_y=h/2

    for cluster_id in range(k):
        cluster_mask=(labels_2d_rgb==cluster_id).astype("uint8")
        area=np.sum(cluster_mask)

        y_coords_cluster,x_coords_cluster=np.where(cluster_mask==1)

        if len(x_coords_cluster)>0:
            cluster_center_x=np.mean(x_coords_cluster)
            cluster_center_y=np.mean(y_coords_cluster)
            distance_to_center=np.sqrt((cluster_center_x-image_center_x)**2+(cluster_center_y-image_center_y)**2)
            score=area/(distance_to_center+1)
            cluster_scores_rgb.append((score,cluster_id))

    cluster_scores_rgb.sort(reverse=True)
    best_cluster_rgb=cluster_scores_rgb[0][1]
    second_best_cluster_rgb=cluster_scores_rgb[1][1]

    # SEGMENTATION MASK RGB
    segmentation_mask_rgb=np.isin(labels_2d_rgb,[best_cluster_rgb,second_best_cluster_rgb]).astype("uint8")

    # EVALUATION RGB
    pred_rgb=(segmentation_mask_rgb>0).astype(np.uint8)
    gt=(binary_mask>0).astype(np.uint8)

    intersection_rgb=int(np.logical_and(pred_rgb,gt).sum())
    union_rgb=int(np.logical_or(pred_rgb,gt).sum())

    iou_rgb=intersection_rgb/union_rgb if union_rgb!=0 else 0
    dice_rgb=(2*intersection_rgb)/(int(pred_rgb.sum())+int(gt.sum())) if (pred_rgb.sum()+gt.sum())!=0 else 0

    total_iou_rgb+=iou_rgb
    total_dice_rgb+=dice_rgb

    # RGB + SPATIAL
    x_coords,y_coords=np.meshgrid(np.arange(w),np.arange(h))
    features=np.dstack((image_rgb,x_coords,y_coords))
    pixel_values=features.reshape((-1,5))
    pixel_values=np.float32(pixel_values)

    _,labels,centers=cv2.kmeans(pixel_values,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    centers_rgb=np.uint8(centers[:,:3])
    segmented_data=centers_rgb[labels.flatten()]
    segmented_image=segmented_data.reshape(image_rgb.shape)

    if i<5:
        plt.imshow(segmented_image)
        plt.title("K-means Segmentation (RGB+Position)")
        plt.axis("off")
        plt.show()

    labels_2d=labels.reshape((h,w))

    # CLUSTER SCORING RGB + SPATIAL
    cluster_scores=[]

    for cluster_id in range(k):
        cluster_mask=(labels_2d==cluster_id).astype("uint8")
        area=np.sum(cluster_mask)

        y_coords_cluster,x_coords_cluster=np.where(cluster_mask==1)

        if len(x_coords_cluster)>0:
            cluster_center_x=np.mean(x_coords_cluster)
            cluster_center_y=np.mean(y_coords_cluster)
            distance_to_center=np.sqrt((cluster_center_x-image_center_x)**2+(cluster_center_y-image_center_y)**2)
            score=area/(distance_to_center+1)
            cluster_scores.append((score,cluster_id))

    cluster_scores.sort(reverse=True)
    best_cluster=cluster_scores[0][1]
    second_best_cluster=cluster_scores[1][1]

    # SEGMENTATION MASK RGB + SPATIAL
    segmentation_mask=np.isin(labels_2d,[best_cluster,second_best_cluster]).astype("uint8")*255

    if i<5:
        plt.imshow(segmentation_mask,cmap="gray")
        plt.title("K-means Binary Mask")
        plt.axis("off")
        plt.show()

    # EVALUATION RGB + SPATIAL
    pred=(segmentation_mask>0).astype(np.uint8)

    intersection=int(np.logical_and(pred,gt).sum())
    union=int(np.logical_or(pred,gt).sum())

    iou=intersection/union if union!=0 else 0
    dice=(2*intersection)/(int(pred.sum())+int(gt.sum())) if (pred.sum()+gt.sum())!=0 else 0

    print(f"IoU: {iou:.4f}, Dice: {dice:.4f}")

    total_iou+=iou
    total_dice+=dice
    count+=1

# FINAL RESULTS
if count>0:
    print("\nFINAL RESULTS")
    print(f"RGB ONLY -> Average IoU: {total_iou_rgb/count:.4f}")
    print(f"RGB ONLY -> Average Dice: {total_dice_rgb/count:.4f}")

    print(f"RGB+SPATIAL -> Average IoU: {total_iou/count:.4f}")
    print(f"RGB+SPATIAL -> Average Dice: {total_dice/count:.4f}")