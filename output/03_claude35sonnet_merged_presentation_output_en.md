# LLM Prompt Recovery Competition

## Agenda

- Competition Overview
- Evaluation Metrics
- Our Approach
  - Data Creation
  - Mean Prompt Search
  - Prompt Modification
  - Topic Word Prediction
- Top 5 Solutions Overview
- Conclusion

## Competition Overview

- Goal: Recover the LLM prompt used to transform a given text
- Dataset: 1300+ original texts paired with rewritten versions from Gemma
- Duration: Feb 28, 2024 - Apr 17, 2024
- Focus: Understanding effective LLM prompting techniques

## Evaluation Metrics

- Using sentence-t5-base to calculate embedding vectors
- Sharpened Cosine Similarity (SCS) with exponent of 3
- SCS attenuates generous scores for incorrect answers
- Exact word matching is crucial for high scores
- Null answers result in errors

## Our Approach: Data Creation

- Created validation data with high correlation to LB
- Sources:
  - Public datasets
  - Self-generated datasets using Gemma, Gemini, GPT4
- Used 40-90 mean prompts to extract correlated data
- Helped create high-scoring mean prompts efficiently

## Our Approach: Mean Prompt Search

- Started with "Rewrite" and searched nltk/wordnet vocabulary
- Focused on human-understandable sentences
- Final mean prompt (score 0.66 on public/private LB):
**'Revise or Revamp this text and phrase to convey the task yet incorporating a plenteous subject, its initial purport while retaining atmosphere yet preserving any intended heartfelt tone, while keeping writing style, making only key improvements to an succinct, dispassionate, and elegant coherence.'**

## Our Approach: Prompt Modification (1/4)

1. Extract "text kind" of original text (Mistral 7B)
   - Predict text kind (e.g., email, essay, letter)
   - Replace "text and phrase" with predicted kind
   - Boosted public LB score by +0.003~0.005
   - Created two versions: with and without text kind

## Our Approach: Prompt Modification (2/4)

2. Extract "tone" difference (Mistral 7B)
   - Predict tone difference between original and rewritten text
   - Insert tone in the middle of the sentence
   - Added "more" before predicted tone
   - Boosted public LB score by +0.015

## Our Approach: Prompt Modification (3/4)

3. Extract "topic" (DeBERTa-v3-large and DeBERTa-v3-base)
   - Extract word differences between original and rewritten text
   - Train models to classify words appearing in rewrite_prompt
   - Insert extracted words into mean prompt
   - Boosted public LB score by +0.01~0.02

## Our Approach: Prompt Modification (4/4)

4. Logic-based post-processing
   - Extract bold words from rewritten_text (7-9% of dataset)
   - Add "song" if "Verse" is in rewritten_text (7-9% of dataset)
   - Boosted public LB score by +0.005

## Our Approach: Topic Word Prediction (1/2)

- Predicted words present in both rewritten_texts and rewrite prompts
- Two methods to handle tokenization issues:
  1. Label only first token of multi-token words
  2. Label all tokens of target word

## Our Approach: Topic Word Prediction (2/2)

- Training details:
  - Used datasets without noise from public dataset
  - Models: deberta-v3-large and deberta-v3-base
  - Post-processing to remove duplicates and words from original_text
  - Explored best thresholds and insertion positions
  - Combined predictions improved CV score from 0.6530 to 0.6902

| Model | Method | CV Score |
|-------|--------|----------|
| Mean prompt | - | 0.6530 |
| deberta-v3-large + deberta-v3-base | method1+method2+exploration | 0.6902 |

## Top 5 Solutions Overview

### Common Themes

1. Use of "lucrarea" token
2. Mean prompts
3. Fine-tuned language models
4. Ensemble approaches
5. Data generation

### 1st Place: Adversarial Attack

- Adversarial string appended to predictions
- Mix of Mistral 7B and Gemma-7B-1.1-it models
- Diverse set of prediction verbs

### 2nd Place: Team Danube

- Optimized mean prompt
- Embedding prediction model
- LLM predictions
- Bruteforce optimization

### 3rd Place

- Mean prompt template
- Fine-tuned MistralForCausalLM for full prompt prediction
- MistralForSequenceClassification as a filter/gate
- MistralForCausalLM for tag prediction
- Clustering models for prompt template selection

### 4th Place: ST5 Tokenizer Attack

- Optimized mean prompt
- Mistral 7B for additional prompt generation
- Exploitation of "lucrarea" token

### 5th Place

- Extensive data generation
- Embedding prediction using Roberta models
- LLM (Mistral 7B) for prompt generation
- Prompt reranking and concatenation
- Exploitation of "lucrarea" token

## Conclusion

- Our approach combined LLM, ML, and logic-based techniques
- Key components:
  - High-quality validation data
  - Optimized mean prompt
  - Multi-faceted prompt modification
  - Advanced topic word prediction
- Achieved significant improvements in evaluation metrics
- Demonstrated the importance of precise prompt engineering

The top solutions in this competition showcased:
- Deep understanding of the underlying sentence-t5 model
- Creative approaches to optimizing for the specific evaluation metric
- Use of adversarial techniques
- Sophisticated model ensembles
- Careful data preparation

These factors were crucial in achieving high scores and advancing the field of LLM prompt recovery.