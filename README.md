# OCR Post Correction for Endangered Language Texts

This repository contains code for models and experiments from the paper "[OCR Post Correction for Endangered Language Texts](https://www.aclweb.org/anthology/2020.emnlp-main.478/)".

Textual data in endangered languages is often found in **formats that are not machine-readable**, including scanned images of paper books. Extracting the text is challenging because there is typically **no annotated data to train an OCR system** for each endangered language. Instead, we focus on post-correcting the OCR output from a general-purpose OCR system. 

:pushpin: In the paper, we present a dataset containing annotations for documents in three critically endangered languages: Ainu, Griko, Yakkha. 

:pushpin: Our model reduces the recognition error rate by 34% on average, over a state-of-the-art OCR system.

This repository contains a sample from our dataset. Get the full dataset [here](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAN__tAC8ehURVRVMVdQQjQzWlBSMkNaOEJKTUpWVFlEQy4u)!

## OCR post-correction 
The goal of OCR post-correction is to automatically correct errors in the text output from an existing OCR system. The existing OCR system is used to obtain a *first pass transcription* of the input image (example below in the endangered language Griko):

<div align="center"><img alt="First pass OCR transcription" width="600px" src="docs/firstpass.png"></div>


The incorrectly recognized characters in the *first pass* are then corrected by the post-correction model.

<div align="center"><img alt="Corrected transcription" width="620px" src="docs/corrected.png"></div>

## Model

As seen in the example above, OCR post-correction is a text-based sequence-to-sequence task. We use a **character-level encoder-decoder architecture with attention** and add several adaptations for the low-resource setting. The paper has all the details!

:pushpin: The model is trained in a **supervised** manner. The training data consists of first pass OCR outputs as the *source* with corresponding manually corrected transcriptions as the *target*.

:pushpin: Some books that contain texts in endangered languages also contain translations of the text in another (usually high-resource) language. We incorporate an additional encoder in the model, with a **multisource** framework, to use the information from these translations if they are available.


## Running Experiments
We provide a sample dataset in this repository and the full dataset from our paper is available [here](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAN__tAC8ehURVRVMVdQQjQzWlBSMkNaOEJKTUpWVFlEQy4u). 

However, this repository can be used to train OCR post-correction models for **documents in any language**!

We provide scripts for both single-source and multisource models:

- The **single-source** model can be used for almost any document and is significantly easier to set up.

- The **multisource** model can only be used if translations are available.

:rocket: If you are using the sample dataset or already have OCR post-correction training data, follow [these steps](docs/postcorrection.md) to run training and evaluation on the model.

:rocket: If you are starting with scanned images or a scanned PDF, follow [the steps here](docs/firstpass.md) to get a *first pass OCR* and create the post-correction dataset.

We'd love to hear about the new datasets and models you build: send us an email at [srijhwan@cs.cmu.edu](mailto:srijhwan@cs.cmu.edu)!

## Citation
Please cite our paper if this repository was useful.
```
@inproceedings{rijhwani-etal-2020-ocr,
    title = "{OCR} {P}ost {C}orrection for {E}ndangered {L}anguage {T}exts",
    author = "Rijhwani, Shruti  and
      Anastasopoulos, Antonios  and
      Neubig, Graham",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.478",
    doi = "10.18653/v1/2020.emnlp-main.478",
    pages = "5931--5942",
}
```



