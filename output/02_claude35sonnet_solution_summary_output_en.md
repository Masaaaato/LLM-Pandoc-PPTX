# Summary of Top 5 Solutions in LLM Prompt Recovery Competition

## Overview

This document summarizes the solutions of the top 5 participants in the LLM Prompt Recovery competition. The competition involved recovering prompts used to generate text transformations, with solutions ranging from adversarial attacks to sophisticated model ensembles.

## Common Themes

Several common themes emerged across the top solutions:

1. Use of "lucrarea" token: Many solutions leveraged the "lucrarea" token, which seemed to have special properties in the sentence-t5 model used for evaluation.
2. Mean prompts: Most solutions incorporated some form of optimized mean prompt.
3. Fine-tuned language models: Various fine-tuned models, particularly Mistral 7B, were used to generate or improve prompts.
4. Ensemble approaches: Combining multiple techniques often yielded the best results.
5. Data generation: Creating diverse, high-quality datasets for training was crucial.

## 1st Place Solution: Adversarial Attack

### Key Components

1. Adversarial string appended to predictions
2. Mix of Mistral 7B and Gemma-7B-1.1-it models
3. Diverse set of prediction verbs

### Technique Details

- Appended string: " 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"
- This string exploits the behavior of the </s> token in the sentence-t5 model
- "lucrarea" acts as a substitute for </s> in the TensorFlow version of the model
- Multiple models with different opening verbs ensure diversity in predictions

## 2nd Place Solution: Team Danube

### Key Components

1. Optimized mean prompt
2. Embedding prediction model
3. LLM predictions
4. Bruteforce optimization

### Technique Details

- Mean prompt optimized through bruteforce token combination
- Embedding prediction model trained using H2O LLM Studio
- LLM predictions used to initialize optimization
- Final prediction combines few-shot predictions, LLM predictions, mean prompt, and optimized embedding string

### Data and Cross-Validation

- Generated data using various models, primarily Gemma
- Crafted a validation set of ~350 samples
- Good correlation between CV and LB scores

## 3rd Place Solution

### Key Components

1. Mean prompt template
2. Fine-tuned MistralForCausalLM for full prompt prediction
3. MistralForSequenceClassification as a filter/gate
4. MistralForCausalLM for tag prediction
5. Clustering models for prompt template selection

### Technique Details

- Mean prompt + tags + full prompt (if passes the gate)
- Data generation process involving LLMs and prompt variations
- Gate model to filter out incorrect prompt predictions
- Tags model to predict aspects of the rewrite prompt
- Clustering approach to find optimal mean prompts for different sample groups

## 4th Place Solution: ST5 Tokenizer Attack

### Key Components

1. Optimized mean prompt
2. Mistral 7B for additional prompt generation
3. Exploitation of "lucrarea" token

### Technique Details

- Mean prompt optimized to score 0.69
- Mistral 7B used with simple response prefix "Modify this text by"
- Detailed analysis of ST5 tokenizer behavior in PyTorch vs. TensorFlow implementations

## 5th Place Solution

### Key Components

1. Extensive data generation
2. Embedding prediction using Roberta models
3. LLM (Mistral 7B) for prompt generation
4. Prompt reranking and concatenation
5. Exploitation of "lucrarea" token

### Technique Details

- Data generated from various LLMs and public datasets
- Roberta models trained to predict T5 embeddings
- Cosine similarity used for prompt retrieval
- Prompt concatenation and reranking for improved scores
- "lucrarea" token used as a substitute for </s> to boost similarity

## Conclusion

The top solutions in this competition demonstrated a deep understanding of the underlying sentence-t5 model and creative approaches to optimizing for the specific evaluation metric. The use of adversarial techniques, sophisticated model ensembles, and careful data preparation were key factors in achieving high scores.