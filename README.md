# Text-labelling

This repo applies Llama3 8B to perform zero-shot multilabel classification on a set of 522 academic abstracts. This set of abstracts are from [CORDIS](https://cordis.europa.eu/), and represent examples of telecommunications research. The topic labels are derived from the classification system used by CORDIS. This is a one-to-many mapping, where each academic abstract can have multiple associated labels.


## Installation

This repo has only been tested on Ubuntu 22.04, python 3.8 and requires at least 16GB of GPU. Note that although the requirements work for my machine, you may need to alter some of the packages to conform to your Nvidia drivers.


```
conda create -n text-labelling python=3.8
```


```
conda activate text-labelling
```

### Installing Llama 3 8B

Follow the instructions at the [Llama 3 git repo](https://github.com/meta-llama/llama3) to install the model. Once installed, copy the `Meta-Llama-3-8B-Instruct/` folder into the root directory of this repo.


### Installing necessary packages

Run the following to install the remaining packages. Due to the nature of setting up Llama 3 8B, this may require tweaking on some sytems.


```
conda env update --file environment.yml
```


## Downloading Data

Data is stored on google drive.

```javascript
mkdir data && cd data
```

```javascript
wget https://drive.google.com/file/d/1YRW6CTs1Pc6gfmzVNST0oP-uP5bqKOXv/view?usp=sharing
```



## Quickstart

For one script to extract relevant data, run Llama 3 on the data, and to evaluate the model’s performance.

```javascript
sh run_all.sh
```


