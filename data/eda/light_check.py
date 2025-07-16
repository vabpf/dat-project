import os
from PIL import Image, ImageStat

def get_image_lightness(image_path):
    """Calculates the average lightness of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        float: The average lightness of the image (0.0 to 255.0 for grayscale,
               or the average of the channel means for color images), or None if
               the file is not a valid image.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale if not already to simplify lightness calculation
            if img.mode != 'L' and img.mode != 'RGB':
                img = img.convert('RGB') # Convert other modes to RGB first

            if img.mode == 'L':
                stat = ImageStat.Stat(img)
                return stat.mean[0]
            elif img.mode == 'RGB':
                stat = ImageStat.Stat(img)
                # Simple average of RGB channel means as a proxy for lightness
                return sum(stat.mean) / len(stat.mean)

    except IOError:
        print(f"Warning: Could not open or read image file: {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None


def explore_directory_lightness(directory_path):
    """Explores the lightness of all images in a directory.

    Args:
        directory_path (str): The path to the directory containing images.

    Returns:
        dict: A dictionary where keys are image filenames and values are their
              average lightness, or None if the directory does not exist.
    """
    print(f"Exploring directory: {directory_path}")
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return None

    image_lightness_data = {}
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        # print(f"Exploring subdirectory: {subdir_path}")        
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            if os.path.isfile(file_path):
                lightness = get_image_lightness(file_path)
                if lightness is not None:
                    image_lightness_data[filename] = lightness

    return image_lightness_data

if __name__ == '__main__':
    # Example usage:
    # Create a dummy directory and some dummy images for testing
    dummy_dir = "test_images"
    os.makedirs(dummy_dir, exist_ok=True)

    # Create a dark grayscale image
    dark_img = Image.new('L', (100, 100), color=30)
    dark_img.save(os.path.join(dummy_dir, "dark_image.png"))

    # Create a bright grayscale image
    bright_img = Image.new('L', (100, 100), color=220)
    bright_img.save(os.path.join(dummy_dir, "bright_image.png"))

    # Create a color image
    color_img = Image.new('RGB', (100, 100), color=(100, 150, 200))
    color_img.save(os.path.join(dummy_dir, "color_image.png"))

    # Explore the lightness of images in the dummy directory
    lightness_results = explore_directory_lightness(dummy_dir)

    if lightness_results:
        print(f"Lightness results for directory: {dummy_dir}")
        for filename, lightness in lightness_results.items():
            print(f"{filename}: {lightness:.2f}")

    # Clean up dummy files and directory
    # os.remove(os.path.join(dummy_dir, "dark_image.png"))
    # os.remove(os.path.join(dummy_dir, "bright_image.png"))
    # os.remove(os.path.join(dummy_dir, "color_image.png"))
    # os.rmdir(dummy_dir)
