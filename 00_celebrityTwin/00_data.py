# !pip install -Uqq duckduckgo_search

# Utils
from pathlib import Path
from typing import List
from time import sleep
from PIL import Image

# Downloads
import requests
import duckduckgo_search as ddg
import logging
from concurrent.futures import ThreadPoolExecutor


#############################################
# HELPER FUNCTIONS
#############################################


def get_image_urls(term: str, max_images=int, VERBOSE=False) -> List[str]:
    """
    Returns the URL of images matching the search term

    Inputs:
    - term: str, the search term
    - max_images: int, the maximum number of images to return
    - VERBOSE: bool, whether to print out the image urls
    """
    if VERBOSE:
        print(f"Getting image urls for {term}...")

    with ddg.DDGS() as ddgs:
        images = ddgs.images(
            term,
            region="wt-wt",
            safesearch="Moderate",
            size=None,
            color=None,
            type_image=None,
            layout="Square",
            license_image=None,
        )
        urls = [image["image"] for image in images]

    if VERBOSE:
        print(f"... gotten: {len(urls)} URLs")

    return urls[:max_images]


def save_images_v1(
    path: Path, subfolder: str, prefix: str, urls: List[str], VERBOSE=False
) -> None:
    """
    Returns None

    Inputs:
    - prefix: str, the prefix to add to the image name
    - urls: list, the list of urls to save
    - subfolder: str, the subfolder to save the images to, based on CWD
    """

    # Save images using the requests and io libraries
    for idx, url in enumerate(urls):
        # cast subfolder as path object and create it if it doesn't exist
        subfolder = Path(subfolder)
        full_path = path / subfolder
        full_path.mkdir(parents=True, exist_ok=True)
        filename = full_path / f"{prefix}_{idx:03d}.jpg"

        # Get the image
        try:
            response = requests.get(url, stream=True, timeout=2)
            response.raise_for_status()

            # Save
            with open(filename, "wb") as out_file:
                out_file.write(response.content)
                sleep(1)

            # Debugging
            if VERBOSE:
                print(f"...saving image {idx} to {filename}")

        except:
            continue
    
    return None

def download_image(idx: int, url: str, full_path: Path, prefix: str) -> None:
    filename = full_path / f"{prefix}_{idx:03d}.jpg"

    try:
        # Get the image
        response = requests.get(url, timeout=2)
        response.raise_for_status()

        # Save
        with open(filename, "wb") as out_file:
            out_file.write(response.content)

        # Log success
        logger.info(f"...saving image {idx} to {filename}")

    except Exception as e:
        # Log any errors
        logger.error(f"Failed to download image {idx} from {url}: {e}")

def save_images(path: Path, subfolder: str, prefix: str, urls: List[str], max_workers=100) -> None:
    # cast subfolder as path object and create it if it doesn't exist
    subfolder = Path(subfolder)
    full_path = path / subfolder
    full_path.mkdir(parents=True, exist_ok=True)

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # For each URL, submit a new task to the executor
        for idx, url in enumerate(urls):
            executor.submit(download_image, idx, url, full_path, prefix)

def convert_images_to_rgb(path: Path, VERBOSE: bool = False):
    """Convert images to RGB mode and delete corrupted images

    Notes
    - p also checks for "palette" image types
    """
    for image_path in path.glob("**/*.jpg"):
        try:
            img = Image.open(image_path)
            if img.mode == "RGBA" or img.mode == "P":
                img = img.convert("RGB")
                new_image_path = image_path.with_suffix(".jpg")
                img.save(new_image_path)
                if VERBOSE:
                    print(f"Converted image: {image_path} to {new_image_path}")
        except:
            if VERBOSE:
                print(f"Corrupted image: {image_path}")
            image_path.unlink()
    return None


###########################
# RUN
###########################


# Get a bunch of training images
def main():
    # Get the current working directory
    CWD = Path.cwd()
    DATAPATH = CWD / Path("data")

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    terms_male = [
        "Johnny Depp", "Jacky Chan", "Keanu Reeves", "Tom Cruise", "Brad Pitt", 
        "Chris Evans", "Jim Carrey", "Will Smith", "Leonardo DiCaprio", "Robert Downey Jr"]
    terms_female = [
        "Scarlett Johansson", "Jennifer Lawrence", "Emma Stone", "Angelina Jolie", "Jennifer Aniston",
        "Michelle Yeoh", "Gal Gadot", "Margot Robbie", "Natalie Portman", "Emma Watson"]
    terms = terms_male + terms_female

    for term in terms:
        urls = get_image_urls(term=term + " headshot", max_images=200, VERBOSE=True)
        save_images(path=DATAPATH, subfolder=term, prefix=term, urls=urls, max_workers=100)

    # Convert images to RGB mode and delete corrupted images before creating the DataBlock
    convert_images_to_rgb(path=DATAPATH, VERBOSE=True)


if __name__ == "__main__":
    main()

###########################
# STUFF
###########################
