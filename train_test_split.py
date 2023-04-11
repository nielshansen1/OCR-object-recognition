import os
import random
import shutil

def split_train_test(source_directory, target_directory, train_percentage=0.8):
    """
    Function to perform train/test split of images of packaging in a source directory.

    Args:
        source_directory (str): Path to the source directory.
        target_directory (str): Path to the target directory where the train/test splits will be created.
        train_percentage (float, optional): Percentage for the train/test split (default is 0.8 for 80% train and 20% test).
    """
    # Create train and test directories
    train_directory = os.path.join(target_directory, "train")
    test_directory = os.path.join(target_directory, "test")
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Collect all images in the source directory and subdirectories
    images = []
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # Only include files with a certain extension (e.g., .jpg)
            if file.endswith(".jpg"):
                source_path = os.path.join(root, file)
                images.append(source_path)

    # Loop through subdirectories in the source directory
    for directory in dirs:
        # Create corresponding train and test subdirectories
        train_subdirectory = os.path.join(train_directory, directory)
        test_subdirectory = os.path.join(test_directory, directory)
        os.makedirs(train_subdirectory, exist_ok=True)
        os.makedirs(test_subdirectory, exist_ok=True)

        # Filter images that belong to the current subdirectory
        subdirectory_images = [image for image in images if directory in image]

        # Calculate the number of images for the train/test split
        total_images = len(subdirectory_images)
        train_images = int(total_images * train_percentage)
        test_images = total_images - train_images

        # Shuffle the images
        random.shuffle(subdirectory_images)

        # Copy train images to the train subdirectory
        for image in subdirectory_images[:train_images]:
            filename = os.path.basename(image)
            target_path = os.path.join(train_subdirectory, filename)
            shutil.copy2(image, target_path)

        # Copy test images to the test subdirectory
        for image in subdirectory_images[train_images:]:
            filename = os.path.basename(image)
            target_path = os.path.join(test_subdirectory, filename)
            shutil.copy2(image, target_path)

    print("Train/Test split completed!")
