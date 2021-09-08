# Acute Myeloid Leukemia FEDn/Studio example 

This example project is a lighter version of Acute Myeloid Leukemia (AML) classification problem addressed in [[1]](#1). Compared to the original work, this project is built with a lighter Convulutional Neural Network and downsampled images to reduce the computation time and resources.  

The purpose of the model is, as described by the original authors: 

"Reliable recognition of malignant white blood cells is a key step in the diagnosis of hematologic malignancies such as Acute Myeloid Leukemia. Microscopic morphological examination of blood cells is usually performed by trained human examiners, making the process tedious, time-consuming and hard to standardise.

We compile an annotated image dataset of over 18,000 white blood cells, use it to train a convolutional neural network for leukocyte classification, and evaluate the network’s performance. The network classifies the most important cell types with high accuracy. 

Our approach holds the potential to be used as a classification aid for examining much larger numbers of cells in a smear than can usually be done by a human expert. This will allow clinicians to recognize malignant cell populations with lower prevalence at an earlier stage of the disease." [[2]](#2).

![Cell image](image.png)

## Participating as a data client in an existing FEDn network 

The below instructions are to set up a local data provider (client in federated learning terminology) to join an exisiting federation (FEDn network). We here assume that you already have access to a deployed FEDn Network (connection information provided by the alliance manager). If this is not the case, to set up a FEDn network either obtain an account in Scaleout Studio or [follow the instructions here](https://github.com/scaleoutsystems/fedn) to set up the network on your own servers.   

Attaching a data client involves three main steps:

1. Setting up the local compute environment / install dependencies. 
2. Stage local training and validation data.
3. Start the *fedn client* pointing to a Reducer endpoint. 

During step 3, a compute package will be downloaded and staged locally, readying the client for executing training and validation requests.

### 1. Set up the client structure and local environment

On you local computer/device, create a folder with the following structure 
```yaml
aml-client
   requirements.txt 
   data/
```
requirements.txt should have the same content as the corresponding file in this repostitory. 
```yaml
tensorflow<2.6.0
pandas
sklearn
```

### 2. Obtain a data partition

Obtain a data partition: 

1. If you are doing this tuturial as part of a workshop you will obtain a download link from the instructor.
2. If not, see instructions below for how to download the raw data and create your own data partitions. 

Unpack the downloaded file and copy the content to the 'data' folder.
```yaml
aml-client
   requirements.txt 
   data/
      --> data_singlets
      --> labels.npy
```

*(If you are not following this tutorial as part of a workshop, see instructions below for how to obtain the data and create your own partitions)*

### 3. Start a client 
You can either start a client natively on Linux/OSX, or use the provided Dockerfile in this repository. 

#### Native client Linux/OSX

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
Install dependencies by the following command:
```bash
$ pip install -r requirements.txt
``` 

4. Access the FEDn UI and download the network configuration file, then place it in the folder (replacing any potential existing client.yaml)

5. Start the client!
```bash
$ fedn run client -in client.yaml --name YOUR_CLIENT_NAME
```

#### Docker

##### Using a prebuilt image

```bash
docker pull scaleoutsystems/fedn-client-aml:latest
```

##### Alternative - download the Dockerfile (or clone this repository), then:

1. Build the docker image:
```bash
docker build . -t aml-client:latest


3. Start a client (edit the path of the volume mounts to provide the absolute path to your local folder)
```
docker run -v /absolute-path-to-this-folder/data/:/app/data:ro -v /absolute-path-to-this-folder/client.yaml:/app/client.yaml scaleoutsystems/fedn-client-aml:latest fedn run client -in client.yaml 
```

## Preparing your own data partitions 

The following instructions are for those that want to prepare their own data partiations from the raw dataset (for example, if you want to change the number of partitions). First, clone this repostitory and install dependencies (requirements.txt). 

### Download the raw data
Download the dataset from:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958

*Note that the download requires a third-party plugin, and that it can take up to a few hours depending on internet connection.*  

### Partion the dataset

place the downloaded folder 'AML-Cytomorphology' in 'dataset/raw', then: 

```bash
python prepare_dataset.py NR_OF_PARTITIONS
```
where NR_OF_PARTITIONS are the number of equal sized splits of the dataset. The script will also downsample the images. To modify this behavior, simply edit prepare_dataset.py. 

## Training, evaluating and serving the model in STACKn / Scaleout Studio

The following sections assumes that you are working in a Scaleout Studio project ([Scaleout Studio](https://www.scaleoutsystems.com])). In your project in Studio, deploy a Jupyter Lab instance from the default 'Jupyter STACKn' image, mounting the project-volume and minio-volume volumes. Then open up a terminal (in your lab instance) and clone this repostitory onto project-volume.

Obtain the raw data, ingest it to your project (for example by uploading it to Minio, then it will be accessible in the Lab session on the 'minio-volume') and create partitions as you see fit (see instructions above). 

### Training a centralized model for a given data partitions 
Follow the instructions in the notebook 'AML_centralized.ipynb'. 

### Plotting a confusion matrix for a given model version in the FEDn model trail 
Follow the instructions in the notebook 'Evalutate.ipynb' (Replace the UUID in the notebook with the desired version from the FEDn model trail. Here we assume that the model trail is accessible on the default path in the Minio instance in your Studio project, if this is not the case,  modify the notebook as needed.)

## References
<a id="1">[1]</a> 
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls [Data set]. The Cancer Imaging Archive." https://doi.org/10.7937/tcia.2019.36f5o9ld

<a id="1">[2]</a> 
Matek, C., Schwarz, S., Spiekermann, K.  et al.  Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks.  Nat Mach Intell   1,  538–544 (2019). https://doi.org/10.1038/s42256-019-0101-9

<a id="1">[3]</a> 
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7
