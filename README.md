Image Processing with Histogram Equalization and CLAHE
Introduction

Image processing is a fundamental aspect of computer vision, enabling the enhancement and analysis of visual information for various applications. Two prominent techniques for improving image contrast are Histogram Equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE). This project provides a comprehensive Python script to apply these techniques to a batch of images, facilitating enhanced visual quality and better feature representation.
Table of Contents

    Project Overview
    Understanding Histogram Equalization
    Understanding CLAHE
    Project Structure
    Prerequisites
    Installation Guide
    Usage Instructions
    Detailed Explanation of the Script
    Examples
    Performance Considerations
    Common Issues and Troubleshooting
    References

Project Overview

This project offers a Python-based solution to enhance image contrast using two methods:

    Histogram Equalization: A technique that adjusts the intensity distribution of an image to utilize the entire range of possible pixel values, thereby enhancing global contrast.

    CLAHE (Contrast Limited Adaptive Histogram Equalization): An advanced method that applies histogram equalization to localized regions of the image, preventing over-amplification of noise and improving local contrast.

The provided script processes all images in a specified input directory and saves the enhanced images to an output directory, allowing users to choose between the two contrast enhancement methods.
Understanding Histogram Equalization

Histogram Equalization is a technique used to improve the global contrast of an image by redistributing the intensity values. It transforms the intensity histogram of the image to approximate a uniform distribution, effectively spreading out the most frequent intensity values. This results in areas of lower contrast gaining a higher contrast, making the image more interpretable.

Advantages:

    Enhances the overall contrast of the image.
    Simple and efficient to implement.

Limitations:

    May amplify noise present in the image.
    Not suitable for images with varying lighting conditions, as it applies a uniform enhancement across the entire image.

Understanding CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) is an improvement over standard histogram equalization. It operates on small regions in the image, called tiles, and applies histogram equalization to each. By limiting the contrast amplification, CLAHE prevents the over-amplification of noise and avoids artifacts in homogeneous areas.

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

Installation Guide

    Clone the Repository:

git clone https://github.com/yourusername/image_processing_project.git
cd image_processing_project

Set Up a Virtual Environment (Optional but Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Required Packages:

    pip install opencv-python-headless numpy matplotlib

    Note: The opencv-python-headless package is used to avoid unnecessary GUI dependencies. If you require GUI functionalities, consider installing opencv-python instead.

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

The process_images.py script is structured as follows:

    Importing Libraries:

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

    os: For handling directory and file operations.
    cv2: OpenCV library for image processing functions.
    numpy: For numerical operations, especially
2. Definición de Funciones

El script define varias funciones para manejar la carga, procesamiento y almacenamiento de imágenes:

a. Función load_image(path):

Esta función carga una imagen desde la ruta especificada utilizando OpenCV.

def load_image(path):
    """Loads an image in BGR format."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at path: {path}")
    return image

    cv2.imread(path): Lee la imagen en formato BGR.
    Verifica si la imagen se ha cargado correctamente; si no, lanza una excepción.

b. Función save_image(path, image):

Guarda la imagen procesada en la ruta especificada.

def save_image(path, image):
    """Saves the image to the specified path."""
    cv2.imwrite(path, image)

    cv2.imwrite(path, image): Escribe la imagen en el disco en la ruta proporcionada.

c. Función display_image(title, image):

Muestra la imagen utilizando Matplotlib para visualización.

def display_image(title, image):
    """Displays an image using matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

    cv2.cvtColor(image, cv2.COLOR_BGR2RGB): Convierte la imagen de BGR a RGB para una visualización correcta con Matplotlib.
    plt.imshow(image_rgb): Muestra la imagen.
    plt.title(title): Establece el título de la ventana de visualización.
    plt.axis('off'): Oculta los ejes para una visualización más limpia.
    plt.show(): Renderiza la imagen en una ventana emergente.

d. Función histogram_equalization(image):

Aplica la ecualización de histograma a una imagen en escala de grises o a cada canal de una imagen en color.

def histogram_equalization(image):
    """Applies histogram equalization to a grayscale or color image."""
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(channel) for channel in channels]
        return cv2.merge(eq_channels)
    else:
        raise ValueError("Unsupported image format.")

    Para imágenes en escala de grises:
        cv2.equalizeHist(image): Aplica la ecualización de histograma.
    Para imágenes en color:
        cv2.split(image): Divide la imagen en sus canales de color individuales.
        Aplica la ecualización de histograma a cada canal por separado.
        cv2.merge(eq_channels): Combina los canales ecualizados en una sola imagen.

e. Función apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):

Aplica CLAHE a una imagen en escala de grises o a cada canal de una imagen en color.

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE to a grayscale or color image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:  # Grayscale image
        return clahe.apply(image)
    elif len(image.shape) == 3:  # Color image
        channels = cv2.split(image)
        clahe_channels = [clahe.apply(channel) for channel in channels]
        return cv2.merge(clahe_channels)
    else:
        raise ValueError("Unsupported image format.")

    cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size): Crea un objeto CLAHE con los parámetros especificados.
    Para imágenes en escala de grises:
        clahe.apply(image): Aplica CLAHE a la imagen.
    Para imágenes en color:
        cv2.split(image): Divide la imagen en canales individuales.
        Aplica CLAHE a cada canal por separado.
        cv2.merge(clahe_channels): Combina los canales procesados en una sola imagen.

f. Función process_images(input_folder, output_folder, method):

Procesa todas las imágenes en la carpeta de entrada utilizando el método especificado y guarda los resultados en la carpeta de salida.

def process_images(input_folder, output_folder, method):
    """Processes all images in the input folder using the specified method and saves them to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            image = load_image(input_path)
            if method == 'histogram_equalization':
                processed_image = histogram_equalization(image)
            elif method == 'clahe':
                processed_image = apply_clahe(image)
            else:
                raise ValueError("Unrecognized method. Use 'histogram_equalization' or 'clahe'.")
            save_image(output_path, processed_image)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
