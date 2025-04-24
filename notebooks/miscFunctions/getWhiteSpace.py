from PIL import Image
import numpy as np

# Function to find dimensions of whitespace from input images
def get_whitespace_padding(image_path, threshold=250):
    img = Image.open(image_path).convert("L")  # convert to grayscale
    img_np = np.array(img)

    # Create a mask of non-white pixels (lower than threshold)
    mask = img_np < threshold

    # Find rows and columns that contain non-white pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get bounds of actual content
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    # Calculate padding on each side
    top_pad = top
    bottom_pad = img_np.shape[0] - bottom
    left_pad = left
    right_pad = img_np.shape[1] - right

    return {
        "top": top_pad,
        "bottom": bottom_pad,
        "left": left_pad,
        "right": right_pad
    }