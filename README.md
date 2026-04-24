# Classification of canine elbow dysplasia using the Tsetlin Machine

This repository contains scripts used in the work for my master's thesis (May 2026). 
The objective of wich is to explore the Tsetlin Machines classification abilities when it comes to canine elbow dysplasia.

## Folder breakdown
### CNN 
The CNN folder contains the jupyter notebook used for running a simple basline CNN for comparison with Tsetlin Machine and CNN model EfficientNet-B4.

### Checks_And_Visualisation
This folder contains any and all code used for visualisation, displaying, checking and validation of the dataset.

### Preprocessing
This folder contains code used for preprosessing of the data. This includes, augmentation, exporting h5 images to PNG format, resizing images in datasets, removing images, automatic rotation to match a reference image and automatic cropping from square to circle.

### Tsetlin_Machine
Lastly this folder contains the pipelines for running Tsetlin Machine with and without optuna optimalisation of hyperparameters. As well as code for extracting results, plots and explainability
