Image Processing with Histogram Equalization and CLAHE
Introduction

Image processing is a fundamental aspect of computer vision, enabling the enhancement and analysis of visual information for various applications. Two prominent techniques for improving image contrast are Histogram Equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE). This project provides a comprehensive Python script to apply these techniques to a batch of images, facilitating enhanced visual quality and better feature representation.
Table of Contents

    Project Overview
    Understanding Histogram Equalization
    Understanding CLAHE
    Project Structure
    Prerequisites
    Usage Instructions
    Detailed Explanation of the Script

Project Overview

This project offers a Python-based solution to enhance image contrast using two methods:

    Histogram Equalization
    CLAHE (Contrast Limited Adaptive Histogram Equalization)

The provided script processes all images in a specified input directory and saves the enhanced images to an output directory, allowing users to choose between the two contrast enhancement methods.

Understanding Histogram Equalization

Histogram Equalization is a technique used to improve the global contrast of an image by redistributing the intensity values. It transforms the intensity histogram of the image to approximate a uniform distribution, effectively spreading out the most frequent intensity values. This results in areas of lower contrast gaining a higher contrast, making the image more interpretable.

![imagen](https://github.com/user-attachments/assets/9879f537-4cfe-4b90-9f1f-5ff881258759)


Advantages:

    Enhances the overall contrast of the image.
    Simple and efficient to implement.

Limitations:

    May amplify noise present in the image.
    Not suitable for images with varying lighting conditions, as it applies a uniform enhancement across the entire image.

Understanding CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) is an improvement over standard histogram equalization. It operates on small regions in the image, called tiles, and applies histogram equalization to each. By limiting the contrast amplification, CLAHE prevents the over-amplification of noise and avoids artifacts in homogeneous areas.

![imagen](https://github.com/user-attachments/assets/9179b47b-8b17-4c5b-adc0-717763ed0650)


Advantages:

    Enhances local contrast, making details in different regions more visible.
    Prevents noise amplification by limiting contrast enhancement.
    Suitable for images with varying lighting conditions.

Limitations:

    Computationally more intensive than standard histogram equalization.
    Requires parameter tuning (e.g., tile size, clip limit) for optimal results.

Project Structure

The project is organized as follows:

image_processing_project/

├── input_images/             # Directory containing original images

├── output_images/            # Directory where processed images will be saved

├── process_images.py         # Main Python script for processing images

└── README.md                 # Project documentation

Prerequisites

Before running the script, ensure that you have the following installed:

    Python 3.x: The script is compatible with Python 3.x versions.
    OpenCV: For image processing operations.
    NumPy: For numerical computations.
    Matplotlib: For displaying images (optional, used in the script for visualization).

Usage Instructions

To process images using the script, use the following command:

python process_images.py <input_directory> <output_directory> <method>

Parameters:

    <input_directory>: Path to the directory containing the original images.
    <output_directory>: Path to the directory where processed images will be saved.
    <method>: The contrast enhancement method to use. Options are:
        histogram_equalization: Applies standard histogram equalization.
        clahe: Applies Contrast Limited Adaptive Histogram Equalization.

Example Usage:

    To apply Histogram Equalization:

python process_images.py input_images/ output_images/ histogram_equalization

To apply CLAHE:

    python process_images.py input_images/ output_images/ clahe

Detailed Explanation of the Script

The script is divided into several key sections:

    Importing Libraries:

    os: Handles file and directory operations.
    cv2: OpenCV library for image processing.
    numpy: For numerical operations, particularly with arrays.
    matplotlib.pyplot: For visualizing images.

Defining Functions:
load_image(path)

    Purpose: Loads an image from the specified path in BGR format.
    Details:
        Uses cv2.imread to read the image.
        If the image cannot be loaded, raises a FileNotFoundError.

save_image(path, image)

    Purpose: Saves the processed image to the specified path.
    Details:
        Uses cv2.imwrite to save the image in its processed state.

show_image(title, image)

    Purpose: Displays the image using Matplotlib for visualization.
    Details:
        Converts the image from BGR to RGB using cv2.cvtColor.
        Uses plt.imshow to render the image with the specified title.
        Removes axis labels for a clean display.

Function 1: histogram_equalization(image)

This is the main function, responsible for applying histogram equalization to either grayscale or color images.
Parameters:

    image (np.ndarray): The input image, which can be either grayscale (2D array) or color (3D array).

Returns:

    np.ndarray: The histogram-equalized image.

How It Works:

    Determine Image Type:
        The function checks the shape of the input image:
            If it has two dimensions (len(image.shape) == 2), it is treated as grayscale.

    For Grayscale Images:
        The function calls equalize_grayscale to perform histogram equalization directly.


    Unsupported Format:
        If the image is neither grayscale nor color, the function raises a ValueError.

Function 2: equalize_grayscale(image)

    Parameters:
        image (np.ndarray): The input grayscale image (2D array).

    Returns:
        np.ndarray: The histogram-equalized grayscale image.

    How It Works:
    
        Compute Histogram:
            The function calculates the histogram of the input image using np.histogram.
            The histogram represents the frequency of each pixel intensity value (0 to 255).
    
        Calculate Cumulative Distribution Function (CDF):
            The cumulative sum of the histogram is calculated using hist.cumsum().
            The CDF indicates the cumulative frequency of pixel intensity values.
    
        Normalize the CDF:
            The CDF is normalized to map intensity values from the input range [0, 255] to the full output range [0, 255].
            This ensures that the histogram stretches across the full range of pixel values.
    
        Handle Flat CDF:
            If the CDF has no range (e.g., all pixel values are the same), the function returns the original image.
    
        Apply Mapping:
            The original pixel values are remapped to new intensity values based on the normalized CDF.
    
        Reconstruct Image:
            The function returns the equalized image as a 2D array.

            ![imagen](https://github.com/user-attachments/assets/1f5d792b-3224-4774-84e5-db89dd50c9a9)

Function: create_custom_clahe

    Parameters:
    image (np.ndarray): The input grayscale image to process. Must be a 2D array.
    clip_limit (float): The threshold for contrast limiting. Higher values result in greater contrast enhancement.
    tile_grid_size (tuple): The grid size for dividing the image into smaller tiles. The image is split into a grid of (rows, columns).
    
    Steps of the Algorithm:

    Input Validation:
        Ensures the input image is grayscale by checking its dimensions (len(image.shape) == 2).

    Tile Division:
        The image is divided into small regions (tiles) based on the tile_grid_size parameter.
        For each tile, the algorithm calculates and enhances its local contrast.

    Histogram Calculation:
        The histogram of pixel intensities for each tile is computed using np.histogram.

    Histogram Clipping:
        Limits the amplification of contrast by capping the histogram values to a predefined clip_limit.
        Excess pixels are redistributed evenly across all intensity levels to maintain the balance.

    CDF Calculation:
        Computes the Cumulative Distribution Function (CDF) of the clipped histogram.
        The CDF is normalized to map pixel values from the input range [0, 255] to the output range [0, 255].

    Pixel Mapping:
        Each pixel in the tile is mapped to a new intensity value based on the normalized CDF.

    Reconstruction:
        The enhanced tiles are combined to reconstruct the full image.


Apply_clahe 

    Parameters:
        image (np.ndarray): The input image, which can be either grayscale or color.
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (tuple): The grid size for dividing the image into tiles.
        
    Behavior:

        Grayscale Images:
            Directly calls create_custom_clahe to apply CLAHE.
    
        Color Images:
            Splits the image into its individual color channels using cv2.split.
            Applies create_custom_clahe to each channel independently.
            Merges the enhanced channels back into a single image using cv2.merge.

    Error Handling:
        If the input image is neither grayscale nor color, raises a ValueError.
  

![imagen](https://github.com/user-attachments/assets/0c673c0a-74bb-42f8-9746-d3ac7126ba6f)

process_images(input_dir, output_dir, method)

    Purpose: Processes all images in the input directory using the specified enhancement method and saves the results to the output directory.
    Details:
        Verifies that the output directory exists; if not, creates it.
        Iterates over each file in the input directory:
            Loads the image using load_image.
            Applies the selected method (histogram_equalization or clahe).
            Saves the processed image using save_image.
            Handles errors gracefully by printing error messages.

Main Entry Point:

    Uses argparse to create a command-line interface.
    Defines three arguments:
        input_dir: Path to the folder containing input images.
        output_dir: Path to the folder where processed images will be saved.
        method: The processing method (histogram_equalization or clahe).
    Parses the arguments and calls the process_images function with the specified parameters.
