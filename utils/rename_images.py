import os
import argparse
from PIL import Image

def convert_and_rename_images(input_folder, output_folder):
    # List all files in the input folder
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
    # Sort files to ensure consistent order
    files.sort()
    
    # Iterate over each file
    for index, filename in enumerate(files, start=1):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        
        # Open the image
        try:
            with Image.open(input_path) as img:
                # Convert image to PNG
                output_filename = f"image_{index:04d}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save the image as PNG
                img.save(output_path, 'PNG')
                print(f"Converted {filename} to {output_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Main function to handle command-line arguments
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert all JPG images in a folder to PNG and rename them.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing JPG images.")
    parser.add_argument('output_folder', type=str, help="Path to the folder to save PNG images.")
    
    # Parse arguments
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")

    # Call the conversion function
    convert_and_rename_images(args.input_folder, args.output_folder)
