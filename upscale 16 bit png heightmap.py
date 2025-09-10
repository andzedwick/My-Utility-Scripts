import numpy as np
import os
from PIL import Image
from scipy.ndimage import zoom

# NOTE: height remains the same, only width is upscaled. This is desired for Unreal Engine. UE5 has 512 as the default heightmap height.
#       While upscaling, if you want to keep the height the same, divide the height by the amount you are multiplying the width by (roughly).

# NOTE: This script takes 16 bit pngs specifically. You can import these from Gaea.
def upscale_heightmap(input_path, output_path, target_size):
    print("Full path to input_file1:", os.path.abspath(input_path))

    # Open the heightmap image as 16-bit
    img = Image.open(input_path)
    original_array = np.array(img, dtype=np.uint16)  # Load as 16-bit integer

    # Debug: Print original array stats
    original_min, original_max = original_array.min(), original_array.max()
    print("Original array shape:", original_array.shape)

    # Calculate the zoom factors for each dimension
    zoom_factors = (target_size[1] / original_array.shape[0], target_size[0] / original_array.shape[1])

    # Use scipy's zoom function to interpolate and upscale the heightmap
    upscaled_array = zoom(original_array, zoom_factors, order=3)  # Cubic interpolation (order=3)
    
    # Normalize to the original range
    upscaled_array = (upscaled_array - upscaled_array.min()) / (upscaled_array.max() - upscaled_array.min())
    upscaled_array = upscaled_array * (original_max - original_min) + original_min

    # Convert the upscaled array back to an image and save as 16-bit PNG
    upscaled_img = Image.fromarray(upscaled_array.astype(np.uint16), mode="I;16")
    upscaled_img.save(output_path, format="PNG")
    print(f"Upscaled heightmap saved to {output_path}")

# Example usage
input_file1 = "./assets/Lakes_Mask.png"  # Replace with your input file path
output_file1 = "./assets/Lakes_Mask 4033x4033.png"  # Replace with your desired output file path
output_file2 = "./assets/Lakes_Mask 8129x8129.png"

input_file2 = "./assets/Lakes_Shore_Mask.png"
output_file3 = "./assets/Lakes_Shore_Mask 4033x4033.png"
output_file4 = "./assets/Lakes_Shore_Mask 8129x8129.png"

input_file3 = "./assets/Rivers_Mask.png"
output_file5 = "./assets/Rivers_Mask 4033x4033.png"
output_file6 = "./assets/Rivers_Mask 8129x8129.png"

input_file4 = "./assets/Rivers_Shore_Mask.png"
output_file7 = "./assets/Rivers_Shore_Mask 4033x4033.png"
output_file8 = "./assets/Rivers_Shore_Mask 8129x8129.png"

input_file5 = "./assets/Output.png"
output_file9 = "./assets/Output 4033x4033.png"
output_file10 = "./assets/Output 8129x8129.png"

target_resolution1 = (4033, 4033)  # Replace with your desired resolution
target_resolution2 = (8129, 8129)

upscale_heightmap(input_file1, output_file1, target_resolution1)
upscale_heightmap(input_file1, output_file2, target_resolution2)

upscale_heightmap(input_file2, output_file3, target_resolution1)
upscale_heightmap(input_file2, output_file4, target_resolution2)

upscale_heightmap(input_file3, output_file5, target_resolution1)
upscale_heightmap(input_file3, output_file6, target_resolution2)

upscale_heightmap(input_file4, output_file7, target_resolution1)
upscale_heightmap(input_file4, output_file8, target_resolution2)

upscale_heightmap(input_file5, output_file9, target_resolution1)
upscale_heightmap(input_file5, output_file10, target_resolution2)