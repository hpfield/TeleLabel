## Text-labelling

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

## Clean Data

```
cd eda
python cordis.py
```


## Label Data

Rather than a list of confirmed labels, we ask the model to output a confidence score for each label so that we can later determine an appropriate threshold. 

Furthermore, we want to know if there is a limit to how many possible labels the model can effectively consider at one time. We have a total of 22 lables, passing the labels to the model one at a time to start with, all the way up to considering all possible labels at once. The more labels we can consider at one time, the faster we can label the full dataset.

```
cd methods/telecoms-topics-classification/
sh llama-3-multilabel-classification.sh
```


## Evaluation

To evaluate the performance of Llama 3 when assigning topic labels to academic abstracts, we consider a range of cutoff points for confidence thresholds and the quantity of possible topics we ask the model to consider at once.

```
cd eval
python llama-3-multilabel-eval.py
```


## Results

Interactive plots at the [repo page](https://hpfield.github.io/text-labelling/)