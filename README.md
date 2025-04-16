# VJ_Technology_Image_Segmentation
# Image Segmentation Project

## Overview

This repository contains the code and documentation for an image segmentation project completed as part of a two-task assignment. The project focuses on preparing a dataset of image masks (Task 1) and training a PyTorch model to perform semantic segmentation on this data (Task 2). The dataset used is a subset of the COCO 2017 training dataset.

## Task 1: Dataset Preparation

### Goal

To generate segmentation masks from the COCO 2017 training annotations for a subset of images.

### Steps

1.  **Subset Selection:**
    * A subset of 8,000 images was selected from the COCO 2017 training dataset (`train2017`).
    * A corresponding annotation file (`subset_train_annotations.json`) was created, containing only the annotation entries relevant to the selected 8,000 images.

2.  **Mask Generation:**
    * A Python function (`create_segmentation_masks`) was developed within a Python script to generate segmentation masks.
    * The function reads the image directory and the subset annotation file.
    * For each image, it iterates through the associated annotations (polygons) and draws filled polygons on a grayscale image.
    * The pixel value used to fill each polygon is derived from the `category_id` of the corresponding object in the annotation (multiplied by 100 for better visualization).
    * The resulting mask images are saved as PNG files in a designated output directory (`masks`).
    * The function also includes the functionality to embed the present `category_ids` into the mask filenames (e.g., `image_001_mask_cats_18_58.png`).

### Code Snippet (Example of Mask Generation Function):


```python
draw.polygon(polygon_points, fill=int(category_id * 100))

### Handling Overlapping Masks

### The code that handles overlapping masks:

mask = np.maximum(mask, np.array(mask_image))

### creating file name

### The code that generates file name:

mask_filename = f"{os.path.splitext(filename)[0]}_mask_cats_{category_id_str}.png"


