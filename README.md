# AML Example Project

### Project description

This example project is a lighter version of Acute Myeloid Leukemia (AML) classification problem addressed in[[1]](#1). This project is built with a lighter Convulution Neural Network and downsampled images to reduce the computation time and resources. The purpose of the model is, as described by the original authors: 

"Reliable recognition of malignant white blood cells is a key step in the diagnosis of hematologic malignancies such as Acute Myeloid Leukemia. Microscopic morphological examination of blood cells is usually performed by trained human examiners, making the process tedious, time-consuming and hard to standardise.

We compile an annotated image dataset of over 18,000 white blood cells, use it to train a convolutional neural network for leukocyte classification, and evaluate the network’s performance. The network classifies the most important cell types with high accuracy. 

Our approach holds the potential to be used as a classification aid for examining much larger numbers of cells in a smear than can usually be done by a human expert. This will allow clinicians to recognize malignant cell populations with lower prevalence at an earlier stage of the disease." [[1]](#1).

### Prerequisites

- Keras 2.0
- sklearn
- matplotlib
- numpy

### Data
![Cell image](https://aml-tjn905630c.studio.k8s-prod.pharmb.io/files/image.png?_xsrf=2%7C0dbbad6d%7C0677356ed7f45001e6613a26bb187d12%7C1589443827)

![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)

## References
<a id="1">[1]</a> 
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls [Data set]. The Cancer Imaging Archive." https://doi.org/10.7937/tcia.2019.36f5o9ld
