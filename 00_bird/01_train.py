# ML
import fastai.vision.all as vision

# import fastbook as fb
# import fastai.vision.widgets as vision_widgets
# import fastai as fa
# import fastcore as fc

# Data
import pandas as pd

# Utils
from datetime import datetime
from pathlib import Path
from IPython.display import clear_output, DisplayHandle
import matplotlib.pyplot as plt

# from typing import List


#############################################
# HELPER FUNCTIONS
#############################################


def update_patch(self, obj):
    """Enable the progress bar to render in VSCode iPython / Jupyter notebooks
    returns None

    Source: https://github.com/microsoft/vscode-jupyter/issues/13163
    """
    clear_output(wait=True)
    self.display(obj)

    return None


def plot_matrix(learner: vision.Learner, logpath: Path, prefix: str):
    """Plots Confusion Matrix and Top Losses
    Returns NONE

    Inputs
    - learner: vision learner object
    - logpath: the path to log the image to
    - prefix: the prefix to add to the image name
    """
    # Init variables
    interp = vision.ClassificationInterpretation.from_learner(learner)

    # Check for folder path
    logpath.mkdir(parents=True, exist_ok=True)

    # Plot top losses as a separate figure and save it
    interp.plot_top_losses(9, nrows=3)
    plt.savefig(logpath / f"{prefix}_topLosses.png", bbox_inches="tight")
    plt.close()

    # Plot confusion matrix as a separate figure and save it
    interp.plot_confusion_matrix(figsize=(6, 6))
    plt.savefig(logpath / f"{prefix}_confusionMatrix.png", bbox_inches="tight")
    plt.close()

    return None


def plot_loss_and_metrics(source: str, logpath: Path, prefix: str):
    """Plot losses and metrics from a CSV file.
    Returns None

    Inputs
    - source: the name of the CSV file. assumes it is in same directory as script.
    - logpath: the path to log the image to
    - prefix: the prefix to add to the image name
    """
    ##################
    ## INITS

    # Variables
    data = pd.read_csv(source)

    # Plot
    fig, ax1 = plt.subplots(figsize=(7, 5))
    fig.suptitle("Losses & Metrics")

    ################
    ## PLOT 1
    ## - training and validation losses on the first y-axis
    ## - add labels and title to the first y-axis

    ax1.plot(data["epoch"], data["train_loss"], label="Training Loss", marker="o")
    ax1.plot(data["epoch"], data["valid_loss"], label="Validation Loss", marker="o")

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    #################
    ## PLOT 2
    ## - create a second y-axis sharing the same x-axis
    ## - plot error rate as a bar plot on the second y-axis
    ## - add values on top of the bars
    ## - add a label to the second y-axis

    ax2 = ax1.twinx()

    bars = ax2.bar(
        x=data["epoch"],
        height=data["error_rate"],
        width=0.4,
        alpha=0.5,
        label="Error Rate",
        color="gray",
    )

    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            x=bar.get_x() + bar.get_width() / 2.0, y=yval, s=round(yval, 4), va="bottom"
        )

    ax2.set_ylabel("Error Rate")

    ##################
    ## CLEANUP
    ## - legends
    ## - save and close

    ax1.legend(loc="upper right")

    plt.savefig(logpath / f"{prefix}_training.png", bbox_inches="tight")
    plt.close()

    return None


def export_model(path: Path, learner: vision.vision_learner, prefix: str):
    """Exports the model to a pickle

    Returns None

    Inputs
    - path: the path to save the model to
    - learner: the learner object
    - prefix: the prefix to add to the model name

    Logic
    - Checks the  callback for CSV logger and removes it
    """
    # Check for CSV Logger
    for cb in learner.cbs:
        if isinstance(cb, vision.CSVLogger):
            learner.remove_cbs(vision.CSVLogger)

    learner.export(path / f"{prefix}_model.pkl")

    return None


#############################################
# RUN
#############################################
if __name__ == "__main__":
    ###################
    ## INIT
    ## - run progress bar fix
    ## - get the current working directory
    ## - check and create folders

    DisplayHandle.update = update_patch

    CWD = Path.cwd()
    DATAPATH = CWD / Path("data")
    LOGPATH = CWD / Path("logs")
    MODELPATH = CWD / Path("models")
    LOGFILE = "training.csv"
    DT = datetime.now().strftime("%y%m%d%H%M")

    for path in [DATAPATH, LOGPATH, MODELPATH]:
        path.mkdir(parents=True, exist_ok=True)

    ###################
    ## LOAD
    ## - create a datablock template
    ## - load data into the datablock

    print("\nINIT...")

    training_images = vision.DataBlock(
        blocks=(vision.ImageBlock, vision.CategoryBlock),
        get_items=vision.get_image_files,
        splitter=vision.RandomSplitter(valid_pct=0.2, seed=42),
        get_y=vision.parent_label,
        item_tfms=vision.RandomResizedCrop(256, min_scale=0.5),
        batch_tfms=vision.aug_transforms(),  # ootb augmentation techniques
    )

    dls = training_images.dataloaders(DATAPATH)

    ###################
    ## TRAIN
    ## - create a learner from loaded data
    ## - run finetune, so only head is retrained

    print("\nTRAINING...")

    learn = vision.vision_learner(
        dls=dls,
        arch=vision.resnet18,
        metrics=vision.error_rate,
        cbs=[
            vision.CSVLogger(fname=LOGFILE)
        ],  # callback to a CSV logger, https://docs.fast.ai/callback.progress.html
    )

    learn.fine_tune(6)

    ###################
    ## EVALUATE
    ## - plot training metrics
    ## - check confusion matrix and errors

    print("\nPLOTTING EVALS")
    plot_loss_and_metrics(source=LOGFILE, logpath=LOGPATH, prefix=DT)
    plot_matrix(learn, logpath=LOGPATH, prefix=DT)

    print("\nEXPORTING")
    export_model(path=MODELPATH, learner=learn, prefix=DT)

    print("\nSCRIPT COMPLETE!")

#############################################
## TICKLERS
#############################################

# [x] Solve Training Logging - https://docs.fast.ai/callback.progress.html, Something to do with callbacks
# [x] Saving - update script to save confusion matrix & top losses to a file <- needs fixing. consider making it one plot
# [x] Logging - update script to save training data to file
# [ ] FIXME - Script - enable running from CLI
