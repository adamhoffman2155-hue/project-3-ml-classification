FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu22.04

LABEL maintainer="Adam Hoffman <adamhoffman21@hotmail.ca>"
LABEL description="Machine Learning Classification for Genomic Data"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install conda packages
RUN conda install -c conda-forge -y \
    pandas=2.0.3 \
    numpy=1.24.3 \
    scipy=1.11.1 \
    scikit-learn=1.3.0 \
    xgboost=2.0.0 \
    lightgbm=4.0.0 \
    matplotlib=3.7.2 \
    seaborn=0.12.2 \
    plotly=5.16.1 \
    jupyter=1.0.0 \
    jupyterlab=4.0.4 \
    ipywidgets=8.0.7 \
    pyyaml=6.0 \
    tqdm=4.65.0 \
    joblib=1.3.1 \
    && conda clean --all --yes

# Copy project files
COPY src/ src/
COPY notebooks/ notebooks/
COPY config/ config/
COPY requirements.txt .

# Install additional pip packages if needed
RUN pip install --no-cache-dir -r requirements.txt || true

# Create data directories
RUN mkdir -p data/{raw,processed,metadata} models/{trained,predictions} results/{metrics,plots,reports} tests

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
