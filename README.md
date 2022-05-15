# Unsupervised Domain Adaptation for Vertebrae Detection and Identification
This repository contains the code I used to produce results for my project thesis at Zurich University of Applied Sciences.
By using a new loss function based on sanity checks, we achieve unsupervised domain adaptation for vertebrae detection and identification.

I extended the work of [McCouat and Glocker, "Vertebrae Detection and Localization in CT with Two-Stage CNNs and Dense Annotations", MICCAI workshop MSKI, 2019](https://arxiv.org/abs/1910.05911) and resued some of the code.

The purpose of this repository is so that other researchers can reproduce the results.

## Setup
Clone this repository and create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

````bash
conda create -n uda-vdi python
conda activate uda-vdi
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
````

Install a tool to extract .rar files:

````bash
sudo apt-get update
sudo apt install unrar
````

## Datasets


### BioMedia Data Set (Source Data Set)
1. Download the data from BioMedia: [https://biomedia.doc.ic.ac.uk/data/spine/](https://biomedia.doc.ic.ac.uk/data/spine/). 
2. In the dropbox package there are collections of spine scans called 'spine-1', 'spine-2', 'spine-3', 
'spine-4' and 'spine-5', download and unzip these files and move all these scans into a directory called
'data/biomedia/training_dataset'. You will also see a zip file called 'spine-test-data', download and unzip this file 
and store it to 'data/biomedia/testing_dataset'.
   
### COVID19-CT Data Set (Target Data Set)
1. Download the dataset from [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6ACUZJ](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6ACUZJ)
by using the script [src/preprocessing/download_harvard_dataset.sh](src/preprocessing/download_harvard_dataset.sh) (Note: replace the API-Token with your personal access token).

```bash
mkdir data/covid19-ct
mkdir data/covid19-ct/subjects
cd data/covid19-ct/subjects
bash ../../../src/preprocessing/download_harvard_dataset.sh
```

Afterwards, unzip the downloaded `dataverse_files.zip` file:
```bash
unzip dataverse_files.zip
rm dataverse_files.zip  # delete this big file
```


Multiple `Subject (xxx).rar` files are extracted - These files can be unzipped as well as split into training and testing data sets using the command:

```bash
mkdir data/covid19-ct/dataset
cd src
python preprocessing/unzip_harvard_covid.py --dataset_path ../data/covid19-ct/subjects --tmp_path ../data/covid19-ct/dataset
```

Copy the labels in the corresponding folder `data/covid19-ct` 

## Detection Module

### Pre-Processing
The downloaded scans have to be divided into smaller patches. Therefore, use the script `src/generate_detection_samples.py`

**BioMedia Data Set:**
```bash
cd src
mkdir ../data/biomedia/samples
mkdir ../data/biomedia/samples/detection
mkdir ../data/biomedia/samples/detection/training
mkdir ../data/biomedia/samples/detection/testing
python generate_detection_samples.py --training_dataset_dir ../data/biomedia/training_dataset --testing_dataset_dir ../data/biomedia/testing_dataset --training_sample_dir ../data/biomedia/samples/detection/training --testing_sample_dir ../data/biomedia/samples/detection/testing --volume_format .nii.gz --label_format .lml
```

**Covid19-CT Data Set:**
```bash
cd src
mkdir ../data/covid19-ct/samples
mkdir ../data/covid19-ct/samples/detection
mkdir ../data/covid19-ct/samples/detection/testing_labeled
python generate_detection_samples.py --testing_dataset_dir ../data/covid19-ct/testing_dataset_labeled --testing_sample_dir ../data/covid19-ct/samples/detection/testing_labeled --volume_format .dcm --label_format .nii.gz
```

### Training
Run the training of the detection module:

```bash
python train.py --epochs 100 --lr 0.001 --batch_size 16 --use_wandb --no_da --use_labeled_tgt
```

### Evaluation

- set `testing_dataset_dir` either to `../data/biomedia/testing_dataset` or `../data/covid19-ct/testing_dataset_labeled`
- When using the `covid19-ct` data set, then set `volume_format`: `.dcm` and `label_format`: `.nii.gz`,
- when using the `biomedia` data set, then set `volume_format`: `.nii.gz` and `label_format`: `.lml`
```bash
python measure.py --testing_dataset_dir <testing_dataset_dir> --volume_format <volume_format> --label_format <label_format> --resume_detection <path/to/detection_model.pth> --ignore_small_masks_detection
```

### Store Detection for UDA
The unsupervised domain adaptation loss of the identification module requires detection samples. Generate these by running:

```bash
python measure.py --testing_dataset_dir ../data/covid19-ct/testing_dataset --volume_format .dcm --label_format .nii.gz --resume_detection <path/to/detection_model.pth>  --without_label --save_detections --ignore_small_masks_detection
```

## Identification Module

### Pre-Processing
The downloaded scans have to be divided into smaller patches. Therefore, use the script `src/generate_identification_samples.py`

**BioMedia Data Set:**
```bash
cd src
mkdir ../data/biomedia/samples/identification
mkdir ../data/biomedia/samples/identification/training
mkdir ../data/biomedia/samples/identification/testing
python generate_identification_samples.py --training_dataset_dir ../data/biomedia/training_dataset --testing_dataset_dir ../data/biomedia/testing_dataset --training_sample_dir ../data/biomedia/samples/identification/training --testing_sample_dir ../data/biomedia/samples/identification/testing --volume_format .nii.gz --label_format .lml
```

```bash
cd src
mkdir ../data/covid19-ct/samples/identification
mkdir ../data/covid19-ct/samples/identification/training
mkdir ../data/covid19-ct/samples/identification/testing
mkdir ../data/covid19-ct/samples/identification/training_labeled
mkdir ../data/covid19-ct/samples/identification/testing_labeled
python generate_identification_samples.py --training_dataset_dir ../data/covid19-ct/training_dataset --testing_dataset_dir ../data/covid19-ct/testing_dataset --training_sample_dir ../data/covid19-ct/samples/identification/training --testing_sample_dir ../data/covid19-ct/samples/identification/testing --without_label --with_detection --volume_format .dcm --label_format .nii.gz
```


## Evaluation

