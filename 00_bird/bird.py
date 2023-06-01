
# %% 

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

# Get the current working directory
cwd = Path.cwd()

# %%
#############################################
# Download images
#############################################

# HELPER FUNCTIONS
def get_image_urls(term: str, max_images = int, VERBOSE = False) -> list:
    '''
    Returns the URL of images matching the search term

    About `with`
    ==========
    - When you see a with statement, the object created in this statement implements a pair of methods known as a context management protocol. 
    - These methods are __enter__() and __exit__().
    - When execution flow enters the with block, __enter__() is invoked on the context object. 
    - When execution flow leaves the with block, for any reason, __exit__() is invoked.
    - This is especially useful when dealing with resources or objects that require proper closing or cleanup after use
    - e.g files, network connections, or database connections.

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

    # Save images using the requests and io libraries
    for idx, url in enumerate(urls):
        
        # Check for subfolder
        if subfolder:
            # cast subfolder as path object and create it if it doesn't exist
            subfolder = Path(subfolder)
            full_path = cwd / subfolder
            full_path.mkdir(parents=True, exist_ok=True)
            filename = full_path / f"{prefix}_{idx:03d}.jpg"
        else:
            filename = cwd / f"{prefix}_{idx:03d}.jpg"
        

        # Get the image
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save
        with open(filename, 'wb') as out_file:
            out_file.write(response.content)

        # Debugging
        if VERBOSE: 
            print(f"...saving image {idx} to {filename}")

#############################################
# RUN
#############################################

imgs = get_image_urls(term = "blue jay", max_images = 1)
save_images(prefix = "blue_jay", urls = imgs, subfolder = "images", VERBOSE = True)

# %%
