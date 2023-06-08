# UI
import streamlit as st

# ML
import fastai.vision.all as vision

# Utils
import os
import re
from time import sleep
from pathlib import Path

##########################
## Load Model
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


model = get_latest_model()
learn_inf = vision.load_learner(model)

##########################
## Design UI
##########################

st.write(
    """
# Bird vs Forest

Upload an image to find out what this Neural Net thinks.
"""
)

# Upload
uploaded_file = st.file_uploader("")
if uploaded_file is not None:
    st.image(uploaded_file, width = 300)
    image = vision.PILImage.create(uploaded_file)
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"Prediction: **{pred.upper()}** | Probability: _{probs[pred_idx]:.04f}_")
else:
    sleep(3)

st.divider()
st.text(f"model version: \n {model}")


"""
####################################################
## NOTES
####################################################

##########################
## PREVIEWING STREAMLIT OVER SSH TUNNEL

To preview streamlit on a local machine while SSHing into a remote machine, run the following command:

`ssh -p 8964 -N -f -L localhost:8501:localhost:8501 root@sshb.jarvislabs.ai`

Here's a breakdown of the command:
-p 8964: This specifies the port used for SSH connections (default is 22, but in your case, Jarvis Labs uses 8964).
-N: This tells SSH that no remote commands will be executed, and is useful for port forwarding scenarios.
-f: This requests SSH to go to background just before command execution, allowing the SSH session to run in the background.
-L localhost:8501:localhost:8501: This sets up the port forwarding. It's saying, "forward connections from localhost:8501 on my local machine to localhost:8501 on the remote machine."

Streamlit Ports
- for L, the port should be the port streamlit is running on. Typically, this is indicated once streamlit is run
- to run streamlit on a fixed port, use `streamlit run <script_name>.py --server.port 8501`
"""
