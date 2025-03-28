import os

# Define the folder containing evaluation images
eval_folder = ".//data//eval"  # Change if needed

# Get sorted list of images
images = sorted([f for f in os.listdir(eval_folder) if f.endswith('.png')])

# Rename images starting from image_0001.png
for index, filename in enumerate(images, start=1):
    new_filename = f"image_{index:04d}.png"
    old_path = os.path.join(eval_folder, filename)
    new_path = os.path.join(eval_folder, new_filename)
    os.rename(old_path, new_path)
    print(f"Renamed {filename} -> {new_filename}")

print("Renaming complete!")
