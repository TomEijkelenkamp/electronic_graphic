import os
import argparse
from PIL import Image

def convert_images_to_black_and_white(input_folder, output_folder):
    """
    Convert all images in the input folder to black and white (grayscale) and save them in the output folder.
    
    Args:
    - input_folder (str): Path to the folder containing the input images.
    - output_folder (str): Path to the folder where black and white images will be saved.
    """
    # Ensure output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Only process files with image extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Open the image file
                with Image.open(input_path) as img:
                    # Convert the image to grayscale
                    bw_img = img.convert("L")
                    
                    # Save the black and white image in the output folder
                    output_path = os.path.join(output_folder, filename)
                    bw_img.save(output_path)
                    print(f"Converted {filename} to black and white.")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
        else:
            print(f"Skipping non-image file: {filename}")

# Main function to parse arguments and run the conversion
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert all images in a folder to black and white.")
    parser.add_argument('input_folder', type=str, help="Path to the input folder containing images.")
    parser.add_argument('output_folder', type=str, help="Path to the output folder where black and white images will be saved.")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the conversion function
    convert_images_to_black_and_white(args.input_folder, args.output_folder)
