# fastai-projects
Projects made during fastai course


## Bird vs Forest
![20230608-005492](https://github.com/lennardong/fastai-projects/assets/29778721/0351cfd0-8e9d-4c78-8af0-4d288c5407f3)

Tuning a neural net to classify bird or forest pictures using a small dataset (~100 images per category) and transfer learning (via ResNet18). 

Implementation Notes:
- Scripts, not Notebooks: refactored code to run as python scripts instead of notebooks
- Explicit, not Implicit: replaced `from fastai.vision import *` with `import... as`
- Code Hygiene: implemented basic linting and formatting
- Deployment: streamlit implementation instead of voila
- Data Augmentation: small dataset augmented via crops, skews, rotates and mirrors.
- Tuning & Diagnosis: use of confusion matrixes and `plot_top_losses` to diagnose dataset for task
