# Using Llama 3 to zero-shot generate topics

## Data

- Separated the data into labelled and unlabelled
- Removed any rows with duplicated descriptions or names, as these pertained to missing values


## Llama 3

### Model
- 8B Instruct
- Run locally
- Uses ~16GB VRAM
- Generates all topics for a given description and title in an average of 2.6 seconds 

### Method
- Configure the model to sequentially accept a system prompt with the abstract title and description and output topic labels for each
- Define the context and expected output format using the system prompt
- Parse output for topic labels

### System Prompt
You are an expert data labeller. Your task is to label an academic abstract with a list of topics which are suitable to associate with that academic abstract. These topics will be used downstream to help academics search for papers they are interested in, so please bear this in mind as you provide topics labels. When you reply, you MUST format your labels as a string array. e.g. ["topic1", "topic2", "topic3"]

Your first abstract to label is:

Title:
{name}

Abstract:
{Description}

## Evaluation
- Currently subjective evaluation by reading a few of the abstracts and asserting that the topics seem reasonable
- Potential future evaluation is to auto-evaluate a few of the generated topics using a more advanced model such as GPT-4

## Improvements
- Refining system prompt: The system might benefit from being explicitly told to include/exclude certain things (acronyms, university names etc)
- Fine-tuning: Llama 3 can be fine tuned on the labelled data to adhere to that style of labelling if desired
- Multi-agent system: The generated labels can be evaluated by an "evaluator" instantiation of Llama 3, in an attempt to ensure that the generated labels are appropriate (essentially giving the model multiple passes over the data but with a system prompt telling it to embody a different perspective)