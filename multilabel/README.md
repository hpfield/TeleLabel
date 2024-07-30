# Multilabel Classification

This section covers the required steps to:

* Create the multilabel classification dataset
* Train, run and evaluate ML methods
* Run and evaluate zero-shot LLM-based classification
* Use ML and LLM methods to label the full dataset
* Compare the performance of each method


## Machine Learning

### Creating the dataset


```javascript
cd data
python create_dataset.py
```


### Training Bert


```javascript
cd train
python bert.py
```


### Evaluating Bert


```javascript
cd eval
python bert.py
```


### Running Bert on full dataset

Running inference with Bert will generate a list of labels for a given csv. The path to the csv must be specified with the `data_file` argument. If no argument is given, this defaults to the test set, which can be used as a reference for formatting data and can be found at `binary/data/test.csv`. If you generated the full dataset, as shown in the project root’s README, you can find the csv file in `binary/data/all_samples.csv`.


```
cd run
python bert.py data_file=path/to/data/file.csv
```


## Zero-shot LLM

### Running llama3-8B on dataset for evaluation

Passes a test set to the LLM and stores the confidence scroes in a json file for evaluation.


```
cd run
torchrun --nproc_per_node 1 llama3-8B_for_eval.py 
```


### Evaluating llama3-8B

Without the `--data_path` parameter, the evaluation will default to analysing the latest labelled data in `binary/outputs/llama3-8B` generated in the previous step. This generates results in `binary/results/llama3-8B` demonstrating the performance of different threshold cutoffs. Plots are generated and stored in `binary/eval/plots/llama3-8B`.


```
cd eval
python llama3-8B.py -a
```


### Labelling full dataset with llama3-8B

Passing a desired metric to the program will take the confidence threshold best suited to optimising that metric from the previous step and apply it to each datapoint, resulting in a binary labelled dataset. Metric argument options are:


```
--metric=Precision
--metric=Recall
--metric=F1_score
--metric=Accuracy
```


The program will default try and label the full dataset in `binary/data/all_samples.csv`, but you may provide your own dataset with `--file_path`.


```
cd run
torchrun --nproc_per_node 1 llama3-8B_for_labelling.py --metric=F1_score
```


### Results

 ![](multilabel/results/llama3-8B/plots/best_thresholds_scatter_plot.png)

## Comparing Bert Vs Llama3-8B

Creates a png plot comparing the two methods across Accuracy, Precision, Recall and F1 Score saved in `results/comparison`.

```javascript
cd eval
python compare_methods.py
```


