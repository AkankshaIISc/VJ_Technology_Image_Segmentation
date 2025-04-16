# VJ_Technology_Image_Segmentation

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

### The code that draws polygon:

draw.polygon(polygon_points, fill=int(category_id * 100))

### The code that handles overlapping masks:

mask = np.maximum(mask, np.array(mask_image))

### The code that generates file name:

mask_filename = f"{os.path.splitext(filename)[0]}_mask_cats_{category_id_str}.png" 

```

## Task 2: Image Segmentation Model Training

### Goal

To train a PyTorch image segmentation model on the dataset prepared in Task 1, achieving accurate segmentation and demonstrating generalization within a 6-hour computational resource constraint.

### Model Architecture

* **UNet:** A UNet architecture was selected due to its effectiveness in segmentation tasks and its relatively efficient design, which is suitable for potential computational constraints.
    * **Encoder:** Downsamples the input image to capture contextual information.
    * **Decoder:** Upsamples the encoded features to create a pixel-wise segmentation map.
    * **Skip Connections:** Facilitate the recovery of fine-grained spatial details.
    * **Note:** The user must implement the specific layers of the UNet architecture within a `UNet` class definition.

### Steps

1.  **Dataset Loading and Preparation:**

    * A custom `SegmentationDataset` class was implemented to handle loading images and their corresponding masks.

    * The `SegmentationDataset` class performs the following:

        * Loads images from a specified directory.
        * Loads annotations from the `subset_train_annotations.json` file.
        * Resizes both images and masks to a consistent size to ensure compatibility for batching.

        ```python
        image = image.resize(self.resize_size, Image.BILINEAR)
        mask = Image.fromarray(mask).resize(self.resize_size, Image.NEAREST)
        ```

        * `image`: A PIL Image representing the input image.
        * `self.resize_size`: The desired output size (e.g., (224, 224)).
        * `Image.BILINEAR`: Interpolation method for resizing images.
        * `mask`: A NumPy array representing the segmentation mask.
        * `Image.fromarray(mask)`: Converts the NumPy array to a PIL Image.
        * `Image.NEAREST`: Interpolation method for resizing masks (preserves sharp boundaries).
        * Applies specified transformations.

        ```python
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        ```

        * `transform`: A `torchvision.transforms.Compose` object containing a sequence of transformations.
        * `transforms.ToTensor()`: Converts a PIL Image to a PyTorch tensor.
        * `transforms.Normalize()`: Normalizes the image pixel values.
        * Returns a tuple of (image tensor, mask tensor) when accessed.

    * `torch.utils.data.DataLoader` was used to create data loaders. Data loaders enable efficient batching, shuffling, and parallel loading of data during training and validation.

2.  **Data Splitting:**

    * The complete dataset was split into training and validation sets.

    ```python
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    ```

    * `full_dataset`: The complete `SegmentationDataset`.
    * `train_size`: The number of samples for the training set.
    * `val_size`: The number of samples for the validation set.
    * `torch.utils.data.random_split`: Function to randomly split a dataset into non-overlapping subsets.

    * This split allows for evaluating the model's generalization performance on unseen data during training, helping to prevent overfitting and tune hyperparameters.

3.  **Model Definition:**

    * The UNet model was defined as a `torch.nn.Module` class.

    * The model definition includes:

        * Convolutional layers for feature extraction.

        ```python
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        ```

        * `nn.Conv2d`: 2D convolutional layer.
        * `in_channels`: Number of input channels.
        * `out_channels`: Number of output channels.
        * `kernel_size`: Size of the convolutional kernel.
        * `padding`: Padding applied to the input.

        * Max pooling or strided convolutions for downsampling (encoder).

        ```python
        nn.MaxPool2d(kernel_size=2, stride=2)
        ```

        * `nn.MaxPool2d`: 2D max pooling layer.
        * `kernel_size`: Size of the pooling window.
        * `stride`: Stride of the pooling operation.

        * Upsampling layers for increasing spatial resolution (decoder).

        ```python
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        ```

        * `nn.ConvTranspose2d`: 2D transposed convolutional layer (for upsampling).

        * Skip connections implemented using tensor concatenation to combine features from the encoder to corresponding layers in the decoder.

        ```python
        torch.cat([x1, x2], dim=1)
        ```

        * `torch.cat`: Concatenates tensors.
        * `x1`, `x2`: Tensors to concatenate.
        * `dim`: The dimension along which to concatenate.

        * A final convolutional layer to produce the pixel-wise segmentation output.

    * **Important:** The specific layers and their configuration within the UNet model need to be implemented by the user based on available resources and desired complexity.

4.  **Loss Function and Optimizer Selection:**

    * `torch.nn.CrossEntropyLoss` was chosen as the loss function. This function is suitable for multi-class classification problems, which is appropriate for semantic segmentation where each pixel is classified into a category.

    * `torch.optim.Adam` was selected as the optimizer for training the model. Adam is a commonly used and effective optimizer that adapts the learning rate for each parameter.

    ```python
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ```

    * `criterion`: The loss function.
    * `nn.CrossEntropyLoss()`: PyTorch's CrossEntropyLoss.
    * `.to(device)`: Moves the loss function to the appropriate device (GPU if available).
    * `optimizer`: The optimizer.
    * `optim.Adam()`: PyTorch's Adam optimizer.
    * `model.parameters()`: The model's trainable parameters.
    * `lr`: The learning rate.

5.  **Training Loop Implementation:**

    * The training loop iterates over the specified number of epochs.

    * Within each epoch:

        * The model is set to training mode.

        ```python
        model.train()
        ```

        * `model`: The UNet model.
        * `model.train()`: Sets the model to training mode (important for layers like BatchNorm or Dropout).

        * The training data loader provides batches of images and masks.

        ```python
        for images, masks in train_loader:
            # ...
        ```

        * `train_loader`: The `DataLoader` for the training set.
        * `images`: A batch of input images.
        * `masks`: A batch of corresponding ground truth masks.

        * The input images and masks are moved to the appropriate device (GPU if available).

        ```python
        images = images.to(device)
        masks = masks.to(device)
        ```

        * `device`: The device to use (e.g., 'cuda' for GPU, 'cpu' for CPU).
        * `.to(device)`: Moves the tensors to the specified device.

        * The optimizer's gradients are reset.

        ```python
        optimizer.zero_grad()
        ```

        * `optimizer`: The optimizer.
        * `optimizer.zero_grad()`: Sets the gradients of all optimized tensors to zero. This is necessary before calculating new gradients.

        * The model makes predictions on the input images.

        ```python
        outputs = model(images)
        ```

        * `model`: The UNet model.
        * `images`: The input images.
        * `outputs`: The model's predictions (the segmentation output).

        * The loss between the model's predictions and the ground truth masks is calculated.

        ```python
        loss = criterion(outputs, masks)
        ```

        * `criterion`: The loss function (`CrossEntropyLoss`).
        * `outputs`: The model's predictions.
        * `masks`: The ground truth masks.
        * `loss`: The calculated loss value.

        * The gradients of the loss with respect to the model's parameters are computed.

        ```python
        loss.backward()
        ```

        * `loss`: The loss value.
        * `loss.backward()`: Computes the gradients of the loss with respect to the model's parameters.

        * The optimizer updates the model's parameters based on the calculated gradients.

        ```python
        optimizer.step()
        ```

        * `optimizer`: The optimizer.
        * `optimizer.step()`: Performs a single optimization step (parameter update).

        * The training loss is accumulated and averaged over the epoch.

        * Optionally, a validation loop can be included within each epoch to evaluate the model's performance on the validation set. This validation loop is similar to the training loop but without gradient calculation or parameter updates.

        * The training loop includes a time check (using Python's `time` module).

        ```python
        import time
        start_time = time.time()
        elapsed_time = time.time() - start_time
        if elapsed_time > 6 * 3600:  # Check for 6-hour limit
            break
        ```

        * `time`: Python's `time` module.
        * `time.time()`: Returns the current time.
        * `elapsed_time`: The time elapsed since the start of training.

### Generalization on Unseen Data

* The validation dataset is used to evaluate the model's ability to generalize to unseen data.
* By monitoring the performance on the validation set, it's possible to detect overfitting (where the model performs well on the training data but poorly on unseen data) and to tune hyperparameters for better generalization.

### Output of Task 2:

* A trained PyTorch segmentation model, saved as a model checkpoint (e.g., `segmentation_model.pth`).


