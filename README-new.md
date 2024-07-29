# Text Labelling: Zero-shot LLM vs Trad-ML

## Installing Llama3-8B

Follow the instructions at the Llama 3 git repo to install the model. Once installed, copy the `Meta-Llama-3-8B-Instruct` folder into the root directory of this repo.


## Downloading Data

Data is stored on google drive.


```
mkdir raw_data
cd raw_data
wget https://drive.google.com/file/d/1YRW6CTs1Pc6gfmzVNST0oP-uP5bqKOXv/view?usp=sharing
```

## Clean Data

Processes the raw data into suitable datasets for binary classification and multilabel downstream tasks.

The `create_full_binary` parameter cleans the entire \~150k samples for later inference. Omit this if only interested in model training and evaluation.


```
cd preprocessing
python process_raw.py create_full_binary=True
```


## Binary Classification


```javascript
cd binary
```


View the README in the `binary` directory for further instruction.


## Multilabel Classification

Multilabel classification approaches can be trained and evaluated without having completed the Binary classification component. However, to perform multilabel classification on the full dataset, Binary classification must have been completed so that only telecoms data is considered.


```javascript
cd multilabel
```


View the README in the `multilabel` directory for further instruction.