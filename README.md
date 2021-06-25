# AML Example Project

This example project is a lighter version of Acute Myeloid Leukemia (AML) classification problem addressed in[[1]](#1). This project is built with a lighter Convulutional Neural Network and downsampled images to reduce the computation time and resources. The purpose of the model is, as described by the original authors: 

"Reliable recognition of malignant white blood cells is a key step in the diagnosis of hematologic malignancies such as Acute Myeloid Leukemia. Microscopic morphological examination of blood cells is usually performed by trained human examiners, making the process tedious, time-consuming and hard to standardise.

We compile an annotated image dataset of over 18,000 white blood cells, use it to train a convolutional neural network for leukocyte classification, and evaluate the network’s performance. The network classifies the most important cell types with high accuracy. 

Our approach holds the potential to be used as a classification aid for examining much larger numbers of cells in a smear than can usually be done by a human expert. This will allow clinicians to recognize malignant cell populations with lower prevalence at an earlier stage of the disease." [[2]](#2).

![Cell image](image.png)

## Attaching a client to an existing Reducer 

Create a folder with the following structure 
```yaml
aml-client
   requirements.txt 
   data/
```
requirements.txt should have the same content as the corresponding file in this repostitory. 
```yaml
tensorflow
pandas
sklearn
```

### Download a data partition

Obtain a data partition (separate workshop instructions): 

Unpack the downloaded file and copy the content to the 'data' folder.
```yaml
aml-client
   requirements.txt 
   data/
      --> data_singlets
      --> labels.npy
```

*(If you are not following this tutorial as part of a workshop, see instructions below for how to obtain the data and create own partitions)*

### Start client

Standing in your created folder: 


1. Create a virtual environment and activate it
```bash
$ python3 -m venv env
$ source env/bin/activate
```

2. Install the fedn client
```bash
$ pip install fedn
```

3. Setup dependencies to set up environment
Install dependncies by the following command:
```bash
$ pip install -r requirements.txt
``` 

4. Download the file and place it in the  folder (replacing any potential existing client.yaml)

5. Start the client!
```bash
$ fedn run client -in client.yaml --name YOUR_CLIENT_NAME
```

## Prepare own partitions for experimentation with FL

Clone this repostitory and follow the below instructions to obtain and partition the raw dataset. Note that the download can take up to a few hours depending on internet connection.  

### Download the data
Download the dataset from:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958


### Partion the dataset

place the downloaded folder 'AML-Cytomorphology' in 'dataset/raw', then: 

```bash
python prepare_dataset.py NR_OF_PARTITIONS
```

## References
<a id="1">[1]</a> 
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls [Data set]. The Cancer Imaging Archive." https://doi.org/10.7937/tcia.2019.36f5o9ld

<a id="1">[2]</a> 
Matek, C., Schwarz, S., Spiekermann, K.  et al.  Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks.  Nat Mach Intell   1,  538–544 (2019). https://doi.org/10.1038/s42256-019-0101-9

<a id="1">[3]</a> 
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7
