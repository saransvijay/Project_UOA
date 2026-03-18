### VisualCue Framework — Capstone Project UOA
# Setup Instructions to run the Web interface code (to see the results in the graphical UI).

Prerequisites
- Python 3.8+
- PyTorch
- See requirements in app.py for full dependencies

# SAM Model Weights
Due to GitHub’s file size limitation of 100 MB per file, the pretrained model weights (e.g., SAM ViT-B with size ~375 MB) could not be uploaded directly to the repository. During development, attempts to upload the model using standard Git and Git Large File Storage (Git LFS) were restricted by repository constraints and size limits.
To address this, the model weights are accessed from the official source:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

- After downloading, place it in the weights/ folder:
weights/
sam_vit_b_01ec64.pth

# Running the Application
- python app.py
