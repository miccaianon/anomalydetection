# Code for implementation of the paper: Anomaly Detection from Extremely Undersampled MRI: An Empirical Analysis towards Automated Acquisition

This code is written with pytorch lightning library, therefore train, validation and test methods are defined in the module.


## Requirements 
Requirements for running this repository are under ``requirements.txt``.

## Dataset and annotations

In this project, we use open sourced fastMRI dataset, which is available [here](https://github.com/facebookresearch/fastMRI).
FLAIR images of the brain portion is used and the binary annotations used in the paper (healthy vs unhealthy) are provided in ``annotations.npy``. 0 is healthy and 1 is unhealthy. The keys are given as ``filename_sliceno.png``.

This repository uses many methods from the fastMRI library, but the accommodate the annotations, there have been some changes made in the DataModule (as well as UNet architecture). The fastmri package used is copied as a subfolder under ``fastmri``.

Furthermore, for the model trained with ground truth coil sensitivity estimates, one needs to obtain coil sensitivities, we did this by using JSENSE reconstruction on fully sampled data provided.


## Define paths

Define the paths mentioned above in ``dirs.yaml``.
- brain_path: Path to where the data is located. Should have train, validation and test split folders under with .h5 files.
- log_path: Where to keep the wandb logs (model checkpoints, images of the model).
- maps_path: Path to previously estimated coil sensitivities.
- anns_path: Path to annotation dictionary file.

## Train a model

There are four different modules are provided in this repository. We use DenseNet as the classifier in the results shown in paper.

To train the model with RSS coil combination on acceleration level 5, run
```bash
python train_rss.py --acceleration 5
```
To train the model with coil combination with previously estimated sensitivities on acceleration level 5, run
```bash
python train_gt_sens.py --acceleration 5
```
To train the model end to end by estimating the sensitivities on acceleration level 5, run
```bash
python train_unet_sens.py --acceleration 5
```
## Testing the model

All train scripts end with testing on the test split of the dataset, for a trained model without training simply execute 

```
model = DenseNetGT(....).load_from_checkpoint(path)
trainer = pl.Trainer(....)
trainer.test(model, data_module= data_module)
```

## Logging

You can further define options ``--project`` and ``--runs`` to customize your wandb logs. 
