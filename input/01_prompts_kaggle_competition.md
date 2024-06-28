You are an assistant helping me create presentation materials. Please output Markdown file (.md) with the presentation content according to the following: 

- Start slide titles with '##'.
- Include a slide for the agenda.
- Use multiple slides to explain things in detail.
- One explanation per slide to expect structured output.
- Include table content on slides if it is important.
- The document introduces a solution to a competition named 'LLM Prompt Recovery'.
- Include a slide summarizing the competition overview and evaluation metrics in detail refferring to the following text.
"""LLM Prompt Recovery
Recover the prompt used to transform a given text


LLM Prompt Recovery

Late Submission
Overview
LLMs are commonly used to rewrite or make stylistic changes to text. The goal of this competition is to recover the LLM prompt that was used to transform a given text.

Start

Feb 28, 2024
Close
Apr 17, 2024
Merger & Entry
Description
NLP workflows increasingly involve rewriting text, but there's still a lot to learn about how to prompt LLMs effectively. This machine learning competition is designed to be a novel way to dig deeper into this problem.

The challenge: recover the LLM prompt used to rewrite a given text. You’ll be tested against a dataset of 1300+ original texts, each paired with a rewritten version from Gemma, Google’s new family of open models.

Evaluation
Evaluation Metric
For each row in the submission and corresponding ground truth, sentence-t5-base is used to calculate corresponding embedding vectors. The score for each predicted / expected pair is calculated using the Sharpened Cosine Similarity, using an exponent of 3. The SCS is used to attenuate the generous score given by embedding vectors for incorrect answers. Do not leave any rewrite_prompt blank as null answers will throw an error."""

- Slides explaining evaluation metrics are required.
- Explain my approach to the competition in detail refferring the following two sources.
"""Source1
12th place soluition🥇 : Modifying mean prompt using LLM, ML and logic-based approach
Many thanks for hosting such an exciting competition!
Great teamwork with @currypurin, @tmhrkt, @masato114, @daikon99 .

The solution with Adversarial Attack was so impressed. Our team couldn’t come up with that idea.

Given the evaluation metrics, we needed to predict the exact words that rewrite prompts have, otherwise score will lose significantly. Therefore, instead of using LLM to predict rewrite prompt itself, we focused on modifying the mean prompt better using LLM, ML and logic-based approach.

Our team’s solution (Private 13th / Public 12th) is as follows:

Create data with high correlation between CV and LB
Search for the high scoring mean prompt
Modify the mean prompt
(1) Add "text kind" of original text (Mistral 7B)
(2) Add "tone" difference between original_text and rewritten_text (Mistral 7B)
(3) Add "topic" difference between original_text and rewritten_text (Deberta-v3)
(4) Add specific keywords from rewritten_text (Logic-based)
The details are as follows:

1. Create validation data with high correlation between CV and LB
Based on the LB score of the submitted mean prompts, we extracted a validation dataset that highly correlates with the hidden test data (Public LB).
We extracted the validation data from:
Public datasets (e.g., https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481811)
Self-generated datasets using Gemma, Gemini, GPT4, etc.
About 40 to 90 mean prompts were used to extract the highly correlated validation data.
The evaluation dataset helped us create high-scoring mean prompts efficiently.
2. Search the high score mean prompt
Start with "Rewrite", we searched all the words in the nltk/wordnet vocabulary set that improve the CV score.

We also had a same approach using sentence-t5 vocabulary set, but the sentence looked meaningless such as </s> and lucrarea as @suicaokhoailang and @dipamc77 mentioned in their solutions. Because its public LB score was low for some reason, we decided to keep the sentence as human-understandable as possible. (We should have followed lucrarea though… lol)

The final mean prompt scored 0.66 on the public and private LB as follows:

'Revise or Revamp this text and phrase to convey the task yet incorporating a plenteous subject, its initial purport while retaining atmosphere yet preserving any intended heartfelt tone, while keeping writing style, making only key improvements to an succinct, dispassionate, and elegant coherence.'
3. Modify the mean prompt
As the sentence of mean prompt above is too general, we added some specific keywords to mean prompt based on original_text and rewritten_text as follows:
(1) Extract "text kind" of original text to mean prompt (Mistral 7B)
The first target is text kind (e.g., email, essay, letter, report, advertisement, etc.)
Mistral7B v1 is used to predict the text kind from the original_text.
text and phrase is replaced by the predicted text kind.
This boosts the public LB score by +0.003~0.005.
Although it boosts the public LB score, we are not confident about how many prompts actually have a “text kind” in the private test dataset because text kind can be omitted (e.g.,“Rewrite this letter more casual” can be “Rewrite this more cussual”). Therefore, we finally selected our final submission as (1) the one with text_kind prediction and (2) the one without text_kind prediction (=text and phrase).
(2) Extract “tone” difference between original_text and rewritten_text (Mistral 7B)
The second target is tone difference (e.g., formal, casual, polite, fun, magical, etc.)
Mistral7B v1 is used to predict it from the difference between original_text and rewritten_text.
Insert the “tone” in the middle of sentence. (The position to insert is optimized by CV score.)
It boost the public LB score by +0.01.
The additional magic keyword was more. We added the word more before the predicted tone (e.g. more polite, more formal, etc.) because Rewrite this text more formal looks more natural than Rewrite this text formal. This also boost +0.005 on the public LB.
(3) Extract “topic” from original_text and rewritten_text (DeBERTa-v3-large and DeBERTa-v3-base)
The third target is topic. The idea is that some of the words in the difference between the original_text and rewritten_text can be included in the rewrite_prompt.
To predict this, we did the following steps:
Extract the difference of the words between original_text and rewritten_text.
Train the models (DeBERTa-v3-large and DeBERTa-v3-base) to classify the words whether it appears in the rewrite_prompt.
Insert the extracted words into the mean prompt.
It boost the public LB score by +0.01~0.02.
(4) Logic-based post processing
Based on the observation of the Gemma7b-it's output, we extracted more specific keywords from the rewritten_text using the following logic.

If the rewritten_text has **(bold markdown), extract the words inside of two ** and add the words to the rewrite_prompt. As far as we probed, the number of the rewritten_text that have ** is about 7-9% of the test dataset.

Example:

rewrite_prompt	rewritten_text
Make the text into a book club's discussion questions	** Book Club Discussion Questions: **\n\n1. What does the text describe as the main point of the article?\n2. What does the text suggest is the purpose of the Pegida group?\n3. Can you summarize the main events that took place a…
Convert the text into a mindfulness meditation guide	** Mindfulness Meditation Guide ** \n\n ** Step 1: Find a comfortable position. ** \n\nSit in a quiet and comfortable space where you can focus without distractions. Your back should be straight and your eyes closed. Rest your hands …
If the word Verse is included in the rewritten_text, add the word “song” to the mean prompt. As far as we probed, the number of the rewritten_text that have Verse is about 7-9% of the test dataset.

It boost the public LB score by +0.005.
"""
"""Source2
Details of ktm part
Data Insight and Generation
Initial Observations
The host provided dataset contained distinctive patterns, notably sentences starting with "Memo" and references to "Oakville".
Through a probing technique, these elements were confirmed to appear in the test dataset's original_text, suggesting a similarity in format between the test data and the provided dataset.
Discoveries
We found original_texts in host provided dataset were generated by AI looking at the following point:

Frequently appeared characteristic word Oakville.
A name of place that does not exist in the real world.
The Sentence gemini unintentionally generated was similar to the original_text in host provided dataset.
From these, we inferred that the original_texts were generated by gemini, and we used gemini to generate about 20000 original_texts.

Rewrite Prompt Generation
Some rewritten_text in the host provided datasets seem to be the leak of rewrite_prompt.
Looking at these prompts, rewrite_prompt contains the information about original_text, which is the text kind.
Example:
original_text: Original text: Memo\n\nTo: All Staff\nFrom: Lynne\nDate:...
Leaked rewrite_prompt: Rephrase this memo to evoke a sense of concern and incorporate the topic "acre", ensuring that the tone reflects worry and apprehension.

In this example, the word memo is the rewrite_prompt.
Therefore, before generating the rewrite_prompt, the text kind contained in the host provided dataset was extracted by gemini.

We also considered that the score could be improved by applying the topic and tone included in the rewrite_prompt. The topics and tones were generated in various formats, including those estimated from the host provided dataset and those generated by claude3.

To generate the Rewrite prompt, the verb, text kind, topic, and tone information was passed to gemini to generate, resulting in the number of the prompt is 50000.

Rewritten Text Generation
Using the original_text and rewrite_prompt generated by the above method, approximately 43000 original text and rewrite prompt pairs were generated by gemma-7b-it. If the generated output contained Sure or **Original Text**, the part of the output was removed using regex.

Topic Word Prediction
Initially, we attempted to predict topics using Mistral-7b-it-v0.2 with supervised fine-tuning, but the outputs were unstable. Therefore, an attempt was made to predict words present in both rewritten_texts and rewrite prompts. Since the tokenizer sometimes split a single word into more than one tokens, we need to align the word and token. To deal with this, two methods were used:

method1
When the word to be predicted is divided into more than one tokens, only the first token is labeled as 1, and the remaining tokens are labeled as -100, referenced by here. During inference, tokens with scores above a threshold in the rewritten_text were decoded.

method2
Labeling all tokens of the target word as 1, not just the first token. During inference, the predictions for tokens within the same word were averaged, and words above a threshold were extracted.

Labeling methods
Original_text tokens were labeled as -100 and excluded from loss calculation.
Stop words were labeled as 0.
Words with differences in capitalization or lemma such as summarized and Summarize were also labeled as 1.
Training
Both methods were trained on datasets without noise data from the public dataset using deberta-v3-large and deberta-v3-base models.
Post process was applied to the predicted word to remove duplicated words and the word also appeared in the original_text.
Then, explored the best thresholds and topic insertion position to maximize the CV score.
CV increase from 0.6530 to 0.6902 by combining the prediction results of both methods into a mean prompt.

model	method	CV
mean prompt	-	0.6530
deberta-v3-large	method1	0.6840
deberta-v3-large + deberta-v3-base	method1	0.6843
deberta-v3-large + deberta-v3-base	method2	0.6834
deberta-v3-large + deberta-v3-base	method1+method2	0.6851
deberta-v3-large + deberta-v3-base	method1+method2+exploration	0.6902
"""


