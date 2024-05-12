#!/bin/bash

python eda/cordis.py

sh methods/telecoms-topics-classification/llama-3-classification.sh

python eval/llama-3-multilabel-eval.py -a