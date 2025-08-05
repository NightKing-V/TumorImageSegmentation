# Use official Python slim image
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install any necessary dependencies here
RUN pip install --no-cache-dir numpy pandas matplotlib nibabel opencv-python h5py jupyter scikit-learn albumentations tqdm

# Default command to run Jupyter Lab
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
