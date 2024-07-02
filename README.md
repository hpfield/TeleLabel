How to setup the environment, run experiments, and where to find specific parts of the project.


To get llama3, clone the repo into models/, then follow the repo’s instructions to download llama3 8B and place the instruct model in the root of the llama3 repo.


## LLM zero-shot


### Always

* Update the LLM parameters in the cfg

### Multilabel

* Data is in the data/processed/cordis-multilabel-telecoms.csv
  * Formatted as a csv with headers ‘text’ and ‘topics’
* Experiments are designed to parameter sweep the number of topics handed to the LLM in a single prompt, predictions are placed into outputs/zero-shot/llama3/multilabel
  * Formatted as json file with headers ‘text’, ‘gt’, ‘labels’
* Evaluation will output both the metric scores at each confidence threshold & number of topics considered per prompt
  * Formatted as csv file with headers ‘threshold’, ‘chunk_size’ and then whatever metrics are specified in the cfg



