#############################################
# Inits
#############################################

# ML
import fastbook as fb
import fastai.vision.all as fav
import fastai as fa
import fastcore as fc

# Utils 
from pathlib import Path
from typing import List

# Get the current working directory
cwd = Path.cwd()

#############################################
# Train the Model
#############################################

# Load data into a datablock
dls = fav.DataBlock(
    blocks = (fav.ImageBlock, fav.CategoryBlock),
    get_items = fav.get_image_files,
    splitter = fav.RandomSplitter(valid_pct=0.2, seed=42),
    get_y = fav.parent_label,
    item_tfms = fav.Resize(256, method='squish'),
).dataloaders(cwd / Path('data'))

learn = fav.vision_learner(dls = dls, arch = fav.resnet18, metrics = fav.error_rate)

def __main__():
    learn.fine_tune(4)
    
