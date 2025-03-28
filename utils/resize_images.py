import os
import argparse
from PIL import Image

def resize_images(input_folder, output_folder, size):
    """
    Resize all images in the input folder to a specified size and save them to the output folder.
    
    Args:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder where resized images will be saved.
    - size (tuple): The target size to which images will be resized (width, height).
    """
    # Ensure output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # List all files in the input folder
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    # Sort files to ensure consistent order
    files.sort()

    # Iterate over each file
    for index, filename in enumerate(files, start=1):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        
        # Open the image
        try:
            with Image.open(input_path) as img:
                # Resize the image to the specified size
                resized_img = img.resize(size, Image.LANCZOS)
                
                # Create output filename based on the index
                output_filename = f"image_{index:04d}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save the resized image as PNG
                resized_img.save(output_path, 'PNG')
                print(f"Resized and saved {filename} as {output_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Main function to handle command-line arguments
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Resize all images in a folder to a specified size and save them.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing images to resize.")
    parser.add_argument('output_folder', type=str, help="Path to the folder to save resized images.")
    parser.add_argument('width', type=int, help="Width of the resized image.")
    parser.add_argument('height', type=int, help="Height of the resized image.")
    
    # Parse arguments
    args = parser.parse_args()

    # Define the size as a tuple
    size = (args.width, args.height)

    # Call the resizing function
    resize_images(args.input_folder, args.output_folder, size)
