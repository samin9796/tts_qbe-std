# Integration of text-to-speech models with a query-by-example spoken term detection system

The system consists of Text-to-Speech (TTS) systems and a query-by-example spoken term detection (QbE-STD) system. The TTS system takes text inputs and generates synthesized audio samples (referred to as queries) that are searched in a unlabelled reference corpus. FastSpeech 2 architecture and Parallel Wavegan vocoder are used to train the TTS system. The search of the queries in the reference corpus is done following this [work](https://github.com/fauxneticien/qbe-std_feats_eval). This repo is also forked from [here](https://github.com/fauxneticien/qbe-std_feats_eval). 

## Usage instructions

### Step 1: Install docker
The script was used to install docker and docker-compose on a fresh instance of Ubuntu 20.04 LTS, based on [DigitalOcean instructions](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04).

```
sudo apt update && \
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && \
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable" && \
sudo apt update && \
apt-cache policy docker-ce && \
sudo apt-get -y install docker-ce && \
sudo curl -L "https://github.com/docker/compose/releases/download/1.28.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
sudo chmod +x /usr/local/bin/docker-compose
```
If you cannot run docker without sudo and getting permission denied error, please follow the instructions from this [link](https://askubuntu.com/questions/477551/how-can-i-use-docker-without-sudo)

### Step 2: Clone this repo
```
git clone https://github.com/samin9796/tts_qbe-std.git
cd tts_qbe-std
```
### Step 3: Set up ```gos-kdl``` dataset locally

```
# Download gos-kdl.zip into qbe-std_feats_eval/tmp directory
wget https://zenodo.org/record/4634878/files/gos-kdl.zip -P tmp/

## Install unzip if necessary
# apt-get install unzip

# Create directory data/raw/datasets/gos-kdl
mkdir -p data/raw/datasets/gos-kdl

# Unzip into directory
unzip tmp/gos-kdl.zip -d data/raw/datasets/gos-kdl
```
### Step 4: Pull the docker image
```
# For extracting wav2vec 2.0 features and running evaluation scripts
docker pull fauxneticien/qbe-std_feats_eval
```

### Step 5: Create the conda environments with the names ```inference``` and ```qbe-std```

If anaconda is not installed in your system, you need to install it first. Otherwise, you can ignore the anaconda installation part. You can follow this [link](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-22-04) or run the commands below to install anaconda.

```
cd /tmp && \
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output anaconda.sh && \
sha256sum anaconda.sh && \
bash anaconda.sh && \
source ~/.bashrc 
```
Once anaconda is installed, you need to follow the step below.

```inference``` environment will contain the packages required for TTS inference. ```qbe-std``` environment is created in this work for future usability.
```
# Create two conda environments with the specified names
conda create -n inference python=3.8 anaconda
conda create -n qbe-std python=3.8 anaconda

```

### Step 6: Install the required packages for TTS system
Activate the ```inference``` environment, install the required packages, and then deactivate the environment

```
conda activate inference
pip install espnet==0.10.6 pyopenjtalk==0.2 pypinyin==0.44.0 parallel_wavegan==0.5.4 gdown==4.4.0 espnet_model_zoo
conda deactivate
```
### Step 7: Install Sox for formatting

```
sudo apt-get install sox
```


### Step 8: Type the query

Step 1-7 has to be performed only once. After successful setup, step 8 is for providing text query as input and getting the audio files as output.
```
# Pipeline integrating a TTS system with a QbE-STD system
bash pipeline.sh
```
After running the script, you will be prompted to type a query at first. Then, the system will return the audio files that contain the query and a comma separated file (CSV) with the similarity scores from the given query and all the reference audio files. These outputs can be found in the ```Output``` directory. 

The original documentation from the [QbE-STD repository](https://github.com/fauxneticien/qbe-std_feats_eval) is kept as it is here to provide better insights about this system. However, only the above-mentioned 7 steps should be followed to run this system. 

## QbE-STD repo documentation

In this project we examine different feature extraction methods ([Kaldi MFCCs](https://kaldi-asr.org/doc/feat.html), [BUT/Phonexia Bottleneck features](https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor), and variants of [wav2vec 2.0](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)) for performing QbE-STD with data from language documentation projects.

A walkthrough of the entire experiment pipeline can be found in [scripts/README.md](https://github.com/fauxneticien/qbe-std_feats_eval/tree/master/scripts). Links to acrhived experiment artefacts uploaded to Zenodo are provided in the [last section](https://github.com/fauxneticien/qbe-std_feats_eval#experiment-data-and-artefacts) of this README file. A description of the analyses based on the data is found in [analyses/README.md](https://github.com/fauxneticien/qbe-std_feats_eval/tree/master/analyses), which also provides links to the [pilot analyses](https://github.com/fauxneticien/qbe-std_feats_eval/blob/master/analyses/xlsr-pilot.md) with a multilingual model, [system evaluations](https://github.com/fauxneticien/qbe-std_feats_eval/blob/master/analyses/mtwv.md), and the [error analysis](https://github.com/fauxneticien/qbe-std_feats_eval/blob/master/analyses/error-analysis.md) (all viewable online as GitHub Markdown).

## Citation

```bibtex
@misc{san2021leveraging,
      title={Leveraging pre-trained representations to improve access to untranscribed speech from endangered languages}, 
      author={San, Nay and Bartelds, Martijn and Browne, Mitchell and Clifford, Lily and Gibson, Fiona and Mansfield, John and Nash, David and Simpson, Jane and Turpin, Myfany and Vollmer, Maria and Wilmoth, Sasha and Jurafsky, Dan},
      year={2021},
      eprint={2103.14583},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Directory structure

The directory structure for this project roughly follows the [Cookiecutter Data Science guidelines](https://drivendata.github.io/cookiecutter-data-science/#directory-structure).

```
├── README.md                    <- This top-level README
├── docker-compose.yml           <- Configurations for launching Docker containers
├── qbe-std_feats_eval.Rproj     <- RStudio project file, used to get repository path using R's 'here' package
├── requirements.txt             <- Python package requirements
├── tmp/                         <- Empty directory to download zip files into, if required
├── data/
│   ├── raw/                     <- Immutable data, not modified by scripts
│   │   ├── datasets/            <- Audio data and ground truth labels placed here
│   │   ├── model_checkpoints/   <- wav2vec 2.0 model checkpoint files placed here
│   ├── interim/                         
│   │   ├── features/            <- features generated by extraction scripts (automatically generated)
│   ├── processed/      
│   │   ├── dtw/                 <- results returned by DTW search (automatically generated)
│   │   ├── STDEval/             <- evaluation of DTW searches (automatically generated)
├── scripts/
│   ├── README.md                <- walkthrough for entire experiment pipeline
│   ├── wav_to_shennong-feats.py <- Extraction script for MFCC and BNF features using the Shennong library
│   ├── wav_to_w2v2-feats.py     <- Extraction script for wav2vec 2.0 features
│   ├── feats_to_dtw.py          <- QbE-STD DTW search using extracted features
│   ├── prep_STDEval.R           <- Helper script to generate files needed for STD evaluation
│   ├── gather_mtwv.R            <- Script to gather Maximum Term Weighted Values generated by STDEval
│   ├── STDEval-0.7/             <- NIST STDEval tool
├── analyses/
│   │   ├── data/                <- Final, post-processed data used in analyses
│   │   ├── mtwv.md              <- MTWV figures and statistics reported in paper
│   │   ├── error-analysis.md    <- Error analyses reported in paper
├── paper/
│   │   ├── ASRU2021.tex         <- LaTeX source file of ASRU paper
│   │   ├── ASRU2021.pdf         <- Final paper submitted to ASRU2021
```

## Experiment data and artefacts

With the exception of raw audio and texts from the Australian language documentation projects (for which we do not have permission to release openly) and those from the [Mavir corpus](http://www.lllf.uam.es/ING/CorpusMavir.html) (which can be obtained from the original distributor, subject to signing their licence agreement), all other data used in and generated by the experiments are available on Zenodo (see [https://zenodo.org/communities/qbe-std_feats_eval](https://zenodo.org/communities/qbe-std_feats_eval)). These are:

- Dataset: Gronings [https://zenodo.org/record/4634878](https://zenodo.org/record/4634878)
- Experiment artefacts:
	- MFCC, BNF and wav2vec 2.0 LibriSpeech 960h features (limited to 50 GB per archive by Zenodo):
		- Archive I (eng-mav, gbb-lg, wbp-jk, and wrl-mb datasets): [https://zenodo.org/record/4635438](https://zenodo.org/record/4635438)
		- Archive II (gbb-pd, gos-kdl, gup-wat, mwf-jm, pjt-sw01, and wrm-pd): [https://zenodo.org/record/4635493](https://zenodo.org/record/4635493)
		- Archive III (w2v2-T11 only; all datasets): [https://zenodo.org/record/4638385](https://zenodo.org/record/4638385)
	- wav2vec 2.0 XLSR53 features:
		- Archive I (eng-mav, gbb-lg, wbp-jk, and wrl-mb datasets): [https://zenodo.org/record/5504371](https://zenodo.org/record/5504371)
		- Archive II (gbb-pd, gos-kdl, gup-wat, mwf-jm, pjt-sw01, and wrm-pd datasets): [https://zenodo.org/record/5504471](https://zenodo.org/record/5504471)
		
	- DTW search and evaluation data: [https://zenodo.org/record/5508217](https://zenodo.org/record/5508217)
