# UI
import streamlit as st

# ML
import fastai.vision.all as vision

# Utils
import os
import re
from time import sleep
from pathlib import Path
from typing import List, Tuple

##########################
## Helper Functions
##########################

def get_latest_model() -> Path:
    """Gets the latest model from a path

    Returns a Path object

    Logic
    - Gets all the files in the folder 'models'
    - Filters for the ones that end with .pkl
    - Sorts them by date
    - Returns the last one
    """
    MODELDIR = Path.cwd() / Path("models")
    files = os.listdir(MODELDIR)
    files = [f for f in files if re.search(r"\.pkl$", f)]
    files = sorted(files)
    
    return MODELDIR / files[-1]

def get_probas(learner, img_file) -> dict:
    ''' Get probabilities for each category
    
    Returns a dictionary of categories and probabilities
    
    Example:
    {'Angelina Jolie': 4.453979272511788e-06,
    'Brad Pitt': 0.00034563025110401213,
    'Chris Evans': 0.006351199001073837,
    ...}

    Inputs:
    - img: a image as byte-object
    - learner: a fastai learner object
    '''

    # Get predictions
    image = vision.PILImage.create(img_file)
    pred, pred_idx, probs = learner.predict(image)
    categories = learner.dls.vocab
    probabilities = dict(zip(categories, map(float, probs)))

    return probabilities


def get_bookends(probas: dict, N:int = 3) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    '''Get top and bottom N candidates
    
    Returns a 2 lists of tuple-pairs of the top and bottom N probabilities

    Example: 
    [('Jacky Chan', 0.5022), ('Will Smith', 0.475581), ('Chris Evans', 0.006351199)]

    Inputs:
    - probas: a dictionary of categories and probabilities
    - N: the number of top and bottom probabilities to return
    '''
    
    # Sort the probabilities
    sorted_probabilities = sorted(probas.items(), key=lambda item: item[1], reverse=True)
    top_N = sorted_probabilities[:N]
    bottom_N = sorted_probabilities[-N:]

    return top_N, bottom_N

##########################
## Design UI
##########################

#############
# Initialize    
model = get_latest_model()
learn_inf = vision.load_learner(model)


#############
# Design UI

st.sidebar.write(
    """

"""
)


with st.sidebar:
    st.title("⭐️ Movie Star Twin")
    st.subheader('''
    Which movie star could be your long-lost twin? Take a photo and find out!
    ''')
    st.divider()
    st.write("""
    ## Details
    This app is built on resnet34 architecture fine-tuned on a dataset of 20 different movie stars (~200 photos each).

    Data was acquired by scraping for headshots of the following actors and actresses on DuckDuckGo:
    
    Male Actors:    
    *Johnny Depp, Jacky Chan, Keanu Reeves, Tom Cruise, Brad Pitt, Chris Evans, Jim Carrey, Will Smith, Leonardo DiCaprio, Robert Downey Jr*
    
    Female Actresses:    
    *Scarlett Johansson, Jennifer Lawrence, Emma Stone, Angelina Jolie, Jennifer Aniston, Michelle Yeoh, Gal Gadot, Margot Robbie, Natalie Portman, Emma Watson*

    ## More Info
    Check out the [GitHub Repo]('https://github.com/lennardong/fastai-projects')
    """)
    st.divider()
    st.text("🛠️ Built by Lenn aka JoyfulTinker")
    

# Upload
uploaded_file = st.camera_input("")
if uploaded_file is not None:
    # st.image(uploaded_file, width = 300)
    image = vision.PILImage.create(uploaded_file)
    probas = get_probas(learn_inf, image)
    top_N, bottom_N = get_bookends(probas)

    # Print top N
    st.write("## Top Candidates")
    for name, proba in top_N:
        st.write(f"{name}: {proba*100:.05f}%")
    
    # Print bottom N
    st.write("## Bottom Candidates")
    for name, proba in bottom_N:
        st.write(f"{name}: {proba*100:.05f}%")

else:
    st.write("## 👆 Say 'cheeeeese!'")
    sleep(3)



"""
####################################################
## NOTES
####################################################

##########################
## PREVIEWING STREAMLIT OVER SSH TUNNEL

To preview streamlit on a local machine while SSHing into a remote machine, run the following command:

`ssh -p 18960 -N -f -L localhost:6006:localhost:6006 root@sshb.jarvislabs.ai`

ssh -p 18960 root@sshb.jarvislabs.ai

Here's a breakdown of the command:
-p 8964: This specifies the port used for SSH connections (default is 22, but in your case, Jarvis Labs uses 8964).
-N: This tells SSH that no remote commands will be executed, and is useful for port forwarding scenarios.
-f: This requests SSH to go to background just before command execution, allowing the SSH session to run in the background.
-L localhost:8501:localhost:8501: This sets up the port forwarding. It's saying, "forward connections from localhost:8501 on my local machine to localhost:8501 on the remote machine."

Streamlit Ports
- for L, the port should be the port streamlit is running on. Typically, this is indicated once streamlit is run
- to run streamlit on a fixed port, use `streamlit run <script_name>.py --server.port 8501`
"""
