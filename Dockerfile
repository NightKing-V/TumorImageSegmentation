# Dockerfile
FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install any necessary dependencies here
RUN pip install numpy pandas matplotlib nibabel scikit-learn opencv-python h5py



# Default command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
