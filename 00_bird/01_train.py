#############################################
# Inits
#############################################

# ML
import fastbook as fb
import fastai.vision.all as vision
import fastai.vision.widgets as vision_widgets
import fastai as fa
import fastcore as fc

# Utils 
import shutil
from pathlib import Path
from typing import List
from PIL import Image
from IPython.display import clear_output, DisplayHandle
from datetime import datetime
import matplotlib.pyplot as plt


#############################################
# HELPER FUNCTIONS
#############################################

def convert_images_to_rgb(folder_path: Path, VERBOSE: bool = False):
    ''' Convert images to RGB mode and delete corrupted images

    Notes
    - p also checks for "palette" image types
    '''
    for image_path in folder_path.glob('**/*.jpg'):
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA' or img.mode == 'P':
                img = img.convert('RGB')
                new_image_path = image_path.with_suffix('.jpg')
                img.save(new_image_path)
                if VERBOSE:
                    print(f"Converted image: {image_path} to {new_image_path}")
        except Exception as e:
            if VERBOSE:
                print(f"Corrupted image: {image_path} - {e}")
            image_path.unlink()

def update_patch(self, obj):
    '''
    Enable the progress bar to render in VSCode iPython / Jupyter notebooks
    
    Source: https://github.com/microsoft/vscode-jupyter/issues/13163
    '''
    clear_output(wait=True)
    self.display(obj)


def debug_model(learner, VERBOSE = False):
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check for folder "plots", if none create it 
    plots_path = Path.cwd() / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    # Create debugging objects 
    interp = vision.ClassificationInterpretation.from_learner(learner)

    # save to file
    if VERBOSE: print(f"...Saving top losses to {plots_path}")
    plt.figure(figsize=(12, 12)) 
    interp.plot_top_losses(5, nrows=2)
    plt.savefig(plots_path / f"{dt}_top_losses.png")

    # save to file 
    if VERBOSE: print(f"...Saving confusion matrix to {plots_path}")
    plt.figure(figsize=(12, 12)) 
    interp.plot_confusion_matrix()
    plt.savefig(plots_path / f"{dt}_confusion_matrix.png")

    # Save most confused to a file
    most_confused = interp.most_confused(min_val=2)
    if VERBOSE: print(f"...Saving most confused to {plots_path}")
    with open(plots_path / f"{dt}_most_confused.txt", 'w') as f:
        for item in most_confused:
            f.write("%s\n" % str(item))


#############################################
# RUN
#############################################

# Run progress bar fix
DisplayHandle.update = update_patch

# Get the current working directory
CWD = Path.cwd()
DATAPATH = CWD / Path('data')

# Convert images to RGB mode and delete corrupted images before creating the DataBlock
convert_images_to_rgb(DATAPATH, VERBOSE = True)

# Load data into a datablock
training_images = vision.DataBlock(
    blocks =  (vision.ImageBlock, vision.CategoryBlock),
    get_items = vision.get_image_files,
    splitter = vision.RandomSplitter(valid_pct=0.2, seed=42),
    get_y = vision.parent_label,
    item_tfms =  vision.RandomResizedCrop(256, min_scale=0.5),
    batch_tfms = vision.aug_transforms()
    )

# Create a dataloader from the datablock
dls = training_images.dataloaders(DATAPATH)

# Train
learn = vision.vision_learner(dls = dls, arch = vision.resnet18, metrics = vision.error_rate)
learn.fine_tune(4)
print(f"Model performance score: {learn.validate()}")

debug_model(learn)

#############################################
# TODO
#############################################

# [x] TODO - solve logging - https://docs.fast.ai/callback.progress.html, Something to do with callbacks
# [] Saving - update script to save confusion matrix & top losses to a file <- needs fixing. consider making it one plot
# [] Logging - update script to save training data to file