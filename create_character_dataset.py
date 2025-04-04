from PIL import Image, ImageDraw, ImageFont
import os
import glob
import random

font_dir = os.path.join(os.environ["WINDIR"], "Fonts")
fonts = glob.glob(font_dir + "\\*.ttf")
# print(fonts[:10])  # Show first 10 fonts
for font in fonts:
    font_name = os.path.basename(font)  # Get the font name from the path
    print(font_name)  # Print the font name

def render_character(character, font_path=None, folder="character_images", font_size=random.randint(25, 100), image_size=(128, 128), text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    """
    Draws a single character onto an image with a given font, size, and color.
    
    Args:
        character (str): The character to render.
        font_path (str): Path to the .ttf font file. If None, uses default font.
        font_size (int): Font size in pixels.
        image_size (tuple): (width, height) of the output image.
        text_color (tuple): (R, G, B) color of the text.
        bg_color (tuple): (R, G, B) background color.
    """
    # Create a blank image with background color
    img = Image.new("RGB", image_size, bg_color)
    draw = ImageDraw.Draw(img)

    # Load the font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()  # Use default system font if no path is given
    except IOError:
        print(f"Error: Font file '{font_path}' not found. Using default font.")
        font = ImageFont.load_default()

    # Get text bounding box and position it randomly in the image
    text_bbox = font.getbbox(character)  # (left, top, right, bottom)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = (random.randint(0, image_size[0] - text_width), random.randint(0, image_size[1] - text_height))

    # position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # Draw the character
    draw.text(position, character, font=font, fill=text_color)

    font_name = os.path.basename(font_path) if font_path else "default_font"
    # Remove extension from font name
    font_name = os.path.splitext(font_name)[0]
    # Remove all illigal characters from the font name
    font_name = "".join(c for c in font_name if c.isalnum() or c in (' ', '_')).rstrip()

    save_path = os.path.join(folder, f"{font_name}_{character}.png")

    # Save the image
    img.save(save_path)


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:',.<>?/`~"

folder = "character_images"
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)

for i in range(10):
    character = random.choice(alphabet)  # Randomly select a character from the alphabet
    font_path = random.choice(fonts)  # Randomly select a font from the available fonts

    render_character(character, font_path=font_path, folder=folder)  # Render the character with the selected parameters
