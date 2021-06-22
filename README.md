# AML Example Project
### Prerequisites

### Install dependencies with

```pip install -r requirements.txt```


### Download the data
Download the dataset from:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958


### Prepare the datasets in partions

place the downloaded folder 'AML-Cytomorphology' in 'dataset/raw'
```
python prepare_dataset.py NR_OF_PARTITIONS
```

### Deploy the model

Start a new project, for example “AML example”

Make sure all resources have been deployed (in particular, click the “Minio”-link and ensure that it responds)

Start a new “Labs” session. It can take a few minutes to initialise it.

In labs, launch the terminal. Now you will check out the AML repository. Since you’re checking out into a non-empty folder, you have to do it this way:

```
git init
git remote add origin https://github.com/scaleoutsystems/aml-example-project.git
git pull --allow-unrelated-histories
```

This repository contains a trained model that is ready to be deployed (models/model.tar). Since we are using the default project structure, we can simple run
```
stackn create model -n aml-model -r minor
```
If you’re not using the default structure, you need to specify the model file to deploy (it should contain your model as well as scripts necessary to run predictions etc).

It can take a minute or so (the model is ~100MB, and needs to be uploaded to the project’s S3 storage).

Again, since we are using the default structure, we can deploy this model using the default Python deployment environment.
```
stackn create deployment -m amp-model -d default-python
```
Again, wait for the deployment to be initialized. You can view follow the progress in the "Deployments"-page.

Once the model is live, you can make requests to your endpoint. In your labs session, go to the folder ``notebooks``, and ``predict.ipynb``. Replace the URL with your endpoint (note that it should end with a forward slash.) You are now ready to run the notebook for your first prediction.

### Project description

This example project is a lighter version of Acute Myeloid Leukemia (AML) classification problem addressed in[[1]](#1). This project is built with a lighter Convulutional Neural Network and downsampled images to reduce the computation time and resources. The purpose of the model is, as described by the original authors: 

"Reliable recognition of malignant white blood cells is a key step in the diagnosis of hematologic malignancies such as Acute Myeloid Leukemia. Microscopic morphological examination of blood cells is usually performed by trained human examiners, making the process tedious, time-consuming and hard to standardise.

We compile an annotated image dataset of over 18,000 white blood cells, use it to train a convolutional neural network for leukocyte classification, and evaluate the network’s performance. The network classifies the most important cell types with high accuracy. 

Our approach holds the potential to be used as a classification aid for examining much larger numbers of cells in a smear than can usually be done by a human expert. This will allow clinicians to recognize malignant cell populations with lower prevalence at an earlier stage of the disease." [[2]](#2).



### Data
![Cell image](image.png)


## References
<a id="1">[1]</a> 
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls [Data set]. The Cancer Imaging Archive." https://doi.org/10.7937/tcia.2019.36f5o9ld

<a id="1">[2]</a> 
Matek, C., Schwarz, S., Spiekermann, K.  et al.  Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks.  Nat Mach Intell   1,  538–544 (2019). https://doi.org/10.1038/s42256-019-0101-9

<a id="1">[3]</a> 
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7
