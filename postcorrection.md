# Train an OCR Post-Correction Model

This document describes how to train a OCR post-correction model. The process is illustrated with the sample dataset found in `sample_dataset/postcorrection`.

Get our full dataset [here](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAN__tAC8ehURVRVMVdQQjQzWlBSMkNaOEJKTUpWVFlEQy4u) or train the model on your own dataset!

If you plan to use your own data, follow the instructions [here](firstpass.md) to prepare the datasets for training.

## Requirements
Python 3+ is required. Pip can be used to install the packages:

```
pip install -r postcorr_requirements.txt
```

## Training

The process of training the post-correction model has two main steps:

* Pretraining with first pass OCR outputs (see `sample_dataset/postcorrection/pretraining`).
* Training with manually corrected transcriptions in a supervised manner (see `sample_dataset/postcorrection/training`).

All the steps for training are compiled in `train_single-source.sh` for a single-source model.

In `train_single-source.sh`, modify the experimental settings in the first few lines to point to the appropriate dataset and desired output folder. It is currently set up to use `sample_dataset`.

Then, simply run
```
bash train_single-source.sh
```

For multisource, use `train_multi-source.sh`.


## Testing

For testing with a single-source model, modify the experimental settings in the first few lines of `test_single-source.sh`. It is currently set up to use `sample_dataset`.

Then run
```
bash test_single-source.sh
```

For multisource, use `test_multi-source.sh`.
