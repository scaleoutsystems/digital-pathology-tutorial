# Acute Myeloid Leukemia classification using a federated Convolutional Neural Network 

The purpose of this workshop is to explore the possibilities of federate learning (FL) through a realistic use-case in digital pathology. We will work with a scenario where a public research dataset is partitioned into a number of subsets, where each subset would corresponds to private data on one hostipital/clinic. We will then use FEDn and STACKn to: 

1. Train a model on data from one clinic (the model one client can obtain in isolation).  
2. Export the weights in the pre-trained network for use as a seed model in a FEDn federation.
3. Deploy a FEDn network, configure local clients, and try to improve the model using federated learning.
4. (Optionally) Deploy the model in production.
5. (Optionally) Deploy a simplistic app that lets a pathologist use it to classify images from a browser interface. 

The local datasets used for each site/clinic/client are created by partitioning a publicly available dataset. If you are doing the workshop led by an instructor, you will take the role of a client setting up a local data node on your own local hardware and join a federation that is deployed by the instructor. You will receive additional instructions for how to obtain your data partition, as well as the configuration file needed to attach to the federation. 

If you are doing the tutorial on your own, you need access to a working deployment of [STACKn](https://github.com/scaleoutsystems/stackn) to do the model and app serving part of the tutorial (Steps 4,5). The federated learning part (1,2,3) can be completed with a deployed [FEDn network](https://github.com/scaleoutsystems/fedn). You will also need to download the raw data and prepare your own data partitions (see instructions below). The data download can take up to a few hours depending on your network connection.    

## The model 

This model we will work with is a slimmed down version of the model developed for the Acute Myeloid Leukemia (AML) classification problem in [[1]](#1). Compared to the original work, here the Convolutional Neural Network (CNN) is slightly simplified, and the images are downssampled in order to reduce the computation time and resources, so that the tutorial can be completed without access to GPU resources (a normal Linux or OSX laptop should be enough in most circumstances).    

The purpose of the model is, as described by the original authors: 

"Reliable recognition of malignant white blood cells is a key step in the diagnosis of hematologic malignancies such as Acute Myeloid Leukemia. Microscopic morphological examination of blood cells is usually performed by trained human examiners, making the process tedious, time-consuming and hard to standardise.

We compile an annotated image dataset of over 18,000 white blood cells, use it to train a convolutional neural network for leukocyte classification, and evaluate the network’s performance. The network classifies the most important cell types with high accuracy. 

Our approach holds the potential to be used as a classification aid for examining much larger numbers of cells in a smear than can usually be done by a human expert. This will allow clinicians to recognize malignant cell populations with lower prevalence at an earlier stage of the disease." [[2]](#2).

![Cell image](image.png)

## Preparing the data partitions 

The following instructions are for those preparing their own data partitions from the raw dataset. This is necessary if your are doing the tutorial on your own. If you take part in an instructor-led workshop, partitions will already be available for download.

First, clone this repostitory and install the dependencies (requirements.txt). 

### Download the raw data
Download the dataset from:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958

*Note that the download requires a third-party plugin, and that it can take up to a few hours depending on internet connection.*  

### Partion the dataset

place the downloaded folder 'AML-Cytomorphology' in the folder 'dataset/raw' in your local clone of this repository, then: 

```bash
python prepare_dataset.py NR_OF_PARTITIONS
```
where NR_OF_PARTITIONS are the desired number of equal sized splits of the dataset. The script will also downsample the images. To modify this behavior, simply edit prepare_dataset.py. 

### Training a centralized model for a given data partition (single clinic)
The next step is to gain some experience training the model using the dataset from a single clinic (one partition). To do this, follow the instructions in the notebook 'single_clinic.ipynb'. 

### Prepare an initial model for use in FEDn
In the final cell of 'single_clinic.ipynb' you will save the weights from the pre-trained single-clinic model as an initial model for use to seed the FEDn federation.  

## Set up a FEDn network 
Again, if you are working in STACKn/Studio as part of a workshop, you will set up the FEDn network in collaboration with the instructor. If you are working on your own, follow the instructions [here](https://github.com/scaleoutsystems/fedn) to deploy a FEDn network on your own hardware.  

Configure FEDn from the UI by uploading the compute package in 'package/package.tar.gz' and the seed file created in the previous step. 

## Participating as a data client in the FEDn network 

The below instructions are to set up a local data provider (client in federated learning terminology) to join the federation (FEDn network).   

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
tensorflow==2.7.2
pandas
scikit-learn
```

### 2. Obtain a data partition

Obtain a data partition: 

1. If you are doing this tuturial as part of a workshop you will obtain a download link from the instructor. Unpack the downloaded file
2. If not, pick a data partition from the set you prepared following the instructions above. 

Copy the content of the partition folder to the 'data' folder.

```yaml
aml-client
   requirements.txt 
   data/
      --> data_singlets
      --> labels.npy
```

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

3. Install dependencies:
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

Build the docker image:
```bash
docker build . -t aml-client:latest
```

Start a client (edit the path of the volume mounts to provide the absolute path to your local folder)
```
docker run -v /absolute-path-to-this-folder/data/:/app/data:ro -v /absolute-path-to-this-folder/client.yaml:/app/client.yaml scaleoutsystems/fedn-client-aml:latest fedn run client -in client.yaml --name YOUR_CLIENT_NAME --secure=True --force-ssl 
```

### Evaulating a given model version in the FEDn model trail 
Follow the instructions in the notebook 'use_fedn_model.ipynb' (Replace the UUID in the notebook with the desired version from the FEDn model trail. Here we assume that the model trail is accessible on the default path in the Minio instance in your Studio project, if this is not the case,  modify the notebook as needed.)
 
### Serving the model (optional)
If you are working in STACKn, you can easily deploy and serve the single clinic model, or any version of the global model using Tensorflow Serving by following the intructions in the notebook 'single_clinic.ipynb'. You can also edit this file to instead/in addition serve any version of the global federated model. 

### Deploying the prediction app
1. Serve the model you want to use with Tensorflow serving (see previous step).
2. Using the STACKn UI, start a notebook mounting the "project-volume". Then, from a terminal, clone this repository onto "project-volume". Edit "app.py" to provide your serving endpoint (make sure that the enpoint is public). From the "Serving" menu, then create a "Dash App", with "Persistent Volume" set to "project-volume" and "Path to folder" set to "aml-example-project/app". 

## References
<a id="1">[1]</a> 
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls [Data set]. The Cancer Imaging Archive." https://doi.org/10.7937/tcia.2019.36f5o9ld

<a id="1">[2]</a> 
Matek, C., Schwarz, S., Spiekermann, K.  et al.  Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks.  Nat Mach Intell   1,  538–544 (2019). https://doi.org/10.1038/s42256-019-0101-9

<a id="1">[3]</a> 
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7
