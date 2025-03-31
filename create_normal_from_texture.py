import tkinter as tk
from tkinter import filedialog
import os

import numpy as np
from PIL import Image
import cv2

from scipy.ndimage import gaussian_filter
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Filename

"""
***DESCRIPTION OF MATHEMATICS IN BELOW PROGRAM***

_________________________________________________________________________________________________________________________________________________
Step 1: Convert RGB Image to a Height Map
_________________________________________________________________________________________________________________________________________________

The first step is to convert a color image (RGB) into a grayscale image (height map).
The idea is that the intensity of each pixel in the grayscale image represents the height value of the surface at that point.

In a height map:
Bright areas represent higher elevation.
Dark areas represent lower elevation.

Mathematical Explanation:
Each pixel in an RGB image consists of three color channels: Red (R), Green (G), and Blue (B). These three channels together encode the color of the pixel.
To convert the RGB image to a grayscale height map, we need to calculate a single value that represents the intensity of the pixel. This intensity corresponds to the height in the height map.

One common way to do this conversion is using the luminosity method, which uses a weighted sum of the RGB channels:

H(x,y)=0.2126⋅R(x,y)+0.7152⋅G(x,y)+0.0722⋅B(x,y)

Where:
H(x,y) is the height at pixel (x,y),
R(x,y), G(x,y), and B(x,y) are the Red, Green, and Blue color values at pixel (x,y),
The coefficients 0.2126, 0.7152, and 0.0722 are based on the human eye's sensitivity to different colors. Green contributes the most, followed by red, and blue has the least impact.

Why this formula?
This formula is based on how our eyes perceive brightness, where green is the most sensitive, and blue is the least sensitive. This gives more realistic results when converting a color image to grayscale.

_________________________________________________________________________________________________________________________________________________
Step 2: Compute the Gradients Using Sobel Filters
_________________________________________________________________________________________________________________________________________________

We need to compute the rate of change (gradient) of the height map in both the X and Y directions. Instead of using simple finite differences, we will use the Sobel filter, which provides a more
accurate and smoother way of estimating derivatives.

These gradients represent the slope of the surface at each pixel and are essential for constructing the normal vectors.

Mathematical Explanation: Sobel Filters
The Sobel filter is a convolution kernel that detects edges and computes gradients in an image. It works by applying a weighted sum of neighboring pixels to emphasize changes in intensity.

We define two Sobel kernels:
1. Sobel X Kernel Sx (for detecting horizontal changes):
    Sx = [[-1, 0, +1]
          [-2, 0, +2]
          [-1, 0, +1]]

    The left side (-1, -2, -1) detects dark-to-bright transitions.
    The right side (+1, +2, +1) detects bright-to-dark transitions.
    The result gives an approximation of the rate of change in the X direction.

2. Sobel Y Kernel Sy (for detecting vertical changes):
    Sy = [[-1, -2, -1]
          [0, 0, 0]
          [+1, +2, +1]]
    
    The top side (-1, -2, -1) detects dark-to-bright transitions.
    The bottom side (+1, +2, +1) detects bright-to-dark transitions.
    The result gives an approximation of the rate of change in the Y direction.

Applying Sobel Filters
To compute the gradient at each pixel, we convolve the height map with these kernels.
    Gradient in X direction Gx (This measures how much the height changes left-to-right):
        Gx = H * Sx
    Gradient in Y direction (This measures how much the height changes top-to-bottom):
        Gy = H * Sy
    
    Here, * denotes convolution, which slides the kernel across the image and computes the weighted sum at each pixel.
    Essentially we take a pixel, take the surrounding pixels to make a 3x3 array of pixels, do a dot product between the Sx or Sy
    gradient kernals (depending on if we are compuyting Gx or Gy), then add all the values in the resulting 3x3 array together and use that
    value as the new value for the pixel. Do this for each pixel in the 2d array and we get the sobel filter result.

_________________________________________________________________________________________________________________________________________________
Step 3: Compute the Surface Normal
_________________________________________________________________________________________________________________________________________________

Now that we have the gradients Gx and Gy, we need to compute the normal vector for each pixel. The surface normal tells us the orientation of the surface at each point.

Mathematical Explanation:
To compute the normal vector, we treat the gradients as components of a 3D vector. The surface normal N=(Nx, Ny, Nz) is computed as:
Nx = -Gx
Ny = -Gy
Nz = 1

Why Nz=1?
Because the normal vector points outward from the surface, and the Z-component is typically set to 1 to give the normal a unit length in 3D space.

Next, we normalize the normal vector so that its length is 1:
Length = sqrt(Gx**2 + Gy**2 + Nz**2)

We then normalize each component:
Nx = -Gx / length  # Normalize X component
Ny = -Gy / length  # Normalize Y component
Nz = Nz / length   # Normalize Z component

Why this formula?
We normalize the normal to have a unit length, ensuring the normal vector has a consistent scale, which is crucial for lighting calculations in 3D graphics.
The X and Y gradients control the direction of the normal, while the Z component is set to 1 to give it outward orientation.
"""

def apply_gaussian_blur_to_normal_map(normal_map_image, sigma=1.0):
    """
    Apply Gaussian Blur to a normal map to smooth out noise.

    :param normal_map_image: The normal map image (PIL Image).
    :param sigma: Standard deviation of the Gaussian kernel (higher value = more blur).
    :return: The smoothed normal map (PIL Image).
    """
    # Convert normal map to numpy array
    normal_map_np = np.array(normal_map_image)

    # Apply Gaussian blur on each channel (assuming the normal map is RGB)
    smoothed_normal_map = np.zeros_like(normal_map_np)
    for c in range(3):  # Apply Gaussian blur to each channel (R, G, B)
        smoothed_normal_map[..., c] = gaussian_filter(normal_map_np[..., c], sigma=sigma)
    
    # Convert back to PIL image
    smoothed_normal_map_image = Image.fromarray(smoothed_normal_map)
    return smoothed_normal_map_image


### SELECT IMAGE FILE ###
root = tk.Tk()
root.withdraw() # Hide the main window

file_types = [
    ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg;*.jpeg"),
    ("BMP files", "*.bmp"),
    ("All files", "*.*")
]

file_path = filedialog.askopenfilename(filetypes=file_types)
file_root, file_ext = os.path.splitext(file_path)
normal_file_path = file_root + "_normal.png"
smoothed_normal_file_path = file_root + "smoothed_normal.png"


if file_path:
    ### STEP 1 ###
    # Load image
    texture = Image.open(file_path)
    #print(im.format, im.size, im.mode)
    #im.show()

    texture = texture.convert("RGB")

    # Convert to NumPy array
    texture_array = np.array(texture).astype(np.float32)

    # Convert to grayscale
    height_map = 0.2126 * texture_array[:, :, 0] + 0.7152 * texture_array[:, :, 1] + 0.0722 * texture_array[:, :, 2]

    # Normalize height values between 0 and 255
    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min()) * 255
    height_map = height_map.astype(np.uint8)

    ### STEP 2 ###
    # Apply Sobel filter to compute gradients
    Gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)  # Gradient in X direction
    Gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)  # Gradient in Y direction

    # Normalize gradients for visualization
    Gx_norm = (Gx - Gx.min()) / (Gx.max() - Gx.min()) * 255
    Gy_norm = (Gy - Gy.min()) / (Gy.max() - Gy.min()) * 255

    ### STEP 3 ###

    # Compute normal components
    Nz = np.ones_like(Gx)  # Set Z to 1
    length = np.sqrt(Gx**2 + Gy**2 + Nz**2)  # Compute length for normalization

    Nx = -Gx / length  # Normalize X component
    Ny = -Gy / length  # Normalize Y component
    Nz = Nz / length   # Normalize Z component

    # Convert to RGB (normalize from [-1,1] to [0,255])
    Nx_rgb = ((Nx + 1) / 2 * 255).astype(np.uint8)
    Ny_rgb = ((Ny + 1) / 2 * 255).astype(np.uint8)
    Nz_rgb = ((Nz + 1) / 2 * 255).astype(np.uint8)

    # Merge into a single normal map image
    normal_map = np.dstack((Nx_rgb, Ny_rgb, Nz_rgb))

    ### DISPLAY ###

    # Save and display the normal map
    normal_image = Image.fromarray(normal_map)
    smoothed_normal_map_image = apply_gaussian_blur_to_normal_map(normal_image, sigma=1.0)
    normal_image.save(normal_file_path)
    smoothed_normal_map_image.save(smoothed_normal_file_path)

    normal_image.show()
    smoothed_normal_map_image.show()