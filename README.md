# bonelab-pytorch-lightning
Tasks, dataset components, loss functions, and utilities for using pytorch lightning for deep learning with medical image data.

## Setup

### 1. Set up the recommended `blptl` conda environment:

CPU:
```commandline
conda create -n blptl -c numerics88 -c conda-forge pytorch torchvision pytorch-lightning torchmetrics scikit-learn numpy pandas scipy matplotlib n88tools vtk simpleitk scikit-image
```

GPU:
```commandline
conda create -n blptl -c numerics88 -c conda-forge pytorch-gpu torchvision pytorch-lightning torchmetrics scikit-learn numpy pandas scipy matplotlib n88tools vtk simpleitk scikit-image
```
### 2. Download and install `bonelab`:

```commandline
# from your main projects folder / wherever you keep your git repos...
# ... with SSH authentication
git clone git@github.com:Bonelab/Bonelab.git
# ... or straight HTTPS
git clone https://github.com/Bonelab/Bonelab.git
# go into the repo
cd Bonelab
# install in editable mode
pip install -e .
```

### 3. Download and install `blpytorchlightning`:

```commandline
# from your main projects folder / wherever you keep your git repos...
# ... with SSH authentication
git clone git@github.com:Bonelab/Bonelab.git
# ... or straight HTTPS
git clone https://github.com/Bonelab/Bonelab.git
# go into the repo
cd Bonelab
# install in editable mode
pip install -e .
```

| Warning: When setting up an environment, you should install things with `conda` first and then `pip`. <br/>If you flip back and forth you'll end up breaking your environment eventually! |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### 4. Create a separate project directory and easily write scripts for training models using the `blptl` environment!