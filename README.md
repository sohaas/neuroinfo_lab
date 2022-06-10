# neuroinfo_lab

This repository is intended for my neuroinformatics lab rotation.
The goal is to transform a given 3D "cube" of meteorological data into a 2D "map" of radar precipitation.

## Requirements

```
tensorflow
numpy
matplotlib
yaml
```

## Data

The dataset is not included in this repository. It belongs in the data/ folder with the following structure:

-   01\_trn\_\*.npy for the training dataset, where \* should be c, t, x, y, respectively
-   01\_tst\_\*.npy for the test dataset, where \* should be c, t, x, y, respectively
-   01\_vld\_\*.npy for the validation dataset, where \* should be c, t, x, y, respectively

## Usage

The training for the DeconvNet can be run with the corresponding bash file or, alternatively, with the following command:

```
python train.py --model <model> --config <config-file>
```
