
# !pip install -Uqq duckduckgo_search

# ML
import fastbook as fb
import fastai.vision.all as fav
import fastai as fa
import fastcore as fc
import fastdownload as fd

# Utils 
import requests
import duckduckgo_search as ddg
from pathlib import Path
from typing import List
from time import sleep

# Get the current working directory
cwd = Path.cwd()

#############################################
# Download images
#############################################

# HELPER FUNCTIONS
def get_image_urls(term: str, max_images = int, VERBOSE = False) -> List[str]:
    '''
    Returns the URL of images matching the search term

    Inputs:
    - term: str, the search term
    - max_images: int, the maximum number of images to return
    - VERBOSE: bool, whether to print out the image urls
    '''
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

    return urls[:max_images]

def save_images(prefix: str, urls: list, subfolder: str, VERBOSE = False) -> None:
    '''
    Returns None

    Inputs:
    - prefix: str, the prefix to add to the image name
    - urls: list, the list of urls to save
    - subfolder: str, the subfolder to save the images to, based on CWD
    '''

    # Save images using the requests and io libraries
    for idx, url in enumerate(urls):
        
        # Check for subfolder
        if subfolder:
            # cast subfolder as path object and create it if it doesn't exist
            subfolder = Path(subfolder)
            full_path = cwd / Path("data") / subfolder
            full_path.mkdir(parents=True, exist_ok=True)
            filename = full_path / f"{prefix}_{idx:03d}.jpg"
        else:
            filename = cwd / Path("data") / f"{prefix}_{idx:03d}.jpg"
        

        # Get the image
        try: 
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save
            with open(filename, 'wb') as out_file:
                out_file.write(response.content)
                sleep(1)
            # Debugging
            if VERBOSE: 
                print(f"...saving image {idx} to {filename}")
        
        except:
            continue

###########################
# RUN
###########################

# Get a bunch of training images
def __main__():
    
    terms = ['bird', 'forest']
    for term in terms:
        urls = get_image_urls(term, max_images = 100, VERBOSE=False)
        save_images(term, urls, subfolder=term, VERBOSE=False)
