# bonelab-pytorch-lightning
Tasks, dataset components, loss functions, and utilities for using pytorch lightning for deep learning with medical image data.

---
## Setup

### 0. Clone this repo
```commandline
# from your main projects folder / wherever you keep your git repos...
# ... with SSH authentication
git clone https://github.com/Bonelab/bonelab-pytorch-lightning.git
# ... or straight HTTPS
git clone git@github.com:Bonelab/bonelab-pytorch-lightning.git
```

### 1. Create a conda environment:

```commandline
conda create -n blptl -c numerics88 -c conda-forge n88tools python=3
conda activate blptl
```

This should work on MacOS and on linux (and on ARC).

### 2. Download and install `Bonelab`:

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

### 3. Install `blpytorchlightning`:

```commandline
# go into the repo
cd bonelab-pytorch-lightning
# install in editable mode
pip install -e .
```

| Warning: When setting up an environment, you should install things with `conda` first and then `pip`.  <br/>If you flip back and forth you'll end up breaking your environment eventually! |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### 4. Use it!

Create a separate project directory and easily write scripts for training models using the `blptl` environment!
You can see some example scripts in https://github.com/Bonelab/hrpqct-knee-segmentation

---
## Documentation

If you want to view the documentation locally:

```commandline
cd docs
make html
```

Then open up `docs/build/html/index.html` in a local browser.

If you add a new module you can update the docs by doing (starting from the root directory):

```commandline
cd docs
sphinx-apidoc -o source ../blpytorchlightning
make html
```

When the repo becomes public the docs can be hosted on a public webpage and locally building them won't be necessary.

---
## How to use

The installed package can be imported at the root level as `blpytorchlightning`.

`blpytorchlightning.dataset_components` - Components for loading, sampling, and transforming data. 
Used for constructing pytorch `Datasets` by composition. 
See `blpytorchlightning.dataset_components.datasets.ComposedDataset` for a generic/flexible example of a composed `Dataset`

`blpytorchlightning.loss_functions` - Custom differentiable loss functions.

`blpytorchlightning.models` - Custom implemented `torch` models e.g. for classification or segmentation.
Should all inherit from `torch.nn.Module`.

`blpytorchlightning.tasks` - Custom implemented `pytorch-lightning` "tasks." 
They abstract away and/or tidy up much of the boilerplate code required for a typical deep learning task, e.g. segmentation.
Should all inherit from `pytorch_lightning.LightningModule`.

`blpytorchlightning.utils` - Utility functions such as for converting level set embeddings to masks, or detecting zero crossings.

---
## Contributing

This project uses unit testing and style guide enforcement. PyCharm CE is highly recommended as a text editor / IDE. 
It will integrate with your `conda` environment and highlight most, if not all, style errors as you go.

If you want to add something:

### 1. Create a new branch

```commandline
git pull
git checkout -b <your-name>/<short-description-of-what-you-are-adding>
```

e.g.

```commandline
git pull
git checkout -b nathan/add-new-vision-transformer-model
```

### 2. Add or modify code.

Type-hinting and numpy-style doc strings are mandatory for all functions and class methods
(otherwise it's too hard for others to figure out how to use them).
If you are adding code that is based on published work then please make sure to put references to the corresponding
paper(s) in your doc-strings.

### 3. Write tests

Currently, unit tests are written using the `unittest` package. 
We may upgrade at some point to use additional packages e.g. `hypothesis`.
Tests are all located in the `tests/` directory, with a directory structure that mirrors that of the source files.
If you add a module, add a corresponding test sub-directory. 
If you add a class or function, add a test case inheriting from  `unittest.TestCase`.
If you modify an existing class or function, add or change the appropriate test methods to the existing test case.

### 4. Run tests

To run all tests at the root level (`bonelab-pytorch-lightning/`):

```commandline
nosetest tests
```
### 5. Ensure style conformance:

First, run `black` at the root level (`bonelab-pytorch-lightning/`):
```commandline
black blpytorchlightning
```

This will automatically fix most style errors.

Second, run `flake8` at the root level (`bonelab-pytorch-lightning/`):

```commandline
flake8 blpytorchlightning
```

If there is no terminal output, there's no problems. If there is terminal output, those are style errors that `black`
couldn't fix and that you should go and correct before trying to merge your code.

### 6. Submit a pull request

The easiest way to do this is from the web interface. Go to https://github.com/Bonelab/bonelab-pytorch-lightning, 
switch to your branch, and click `Contribute` > `Open pull request`. 
If the tests and style checks pass it can be merged.

[Full explanation on GitHub Docs.](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
