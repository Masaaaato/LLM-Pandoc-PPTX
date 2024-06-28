I would like to create a document summarizing the solutions of the top 5 participants in a competition. Please output it as a Markdown file (.md). Summarize the similarities and differences of the solutions according to the following guidelines:

- Divide the considerations into several sections, starting each with a title beginning with ##.
- Provide as detailed an explanation as possible.
- Use multiple sections to explain things in detail.
- One explanation per section to expect structured output.
- Include table content on sections if it is important.
- The solutions are as follows:
"""1st solution
1st place solution: adversarial attack
Inference notebook: https://www.kaggle.com/code/suicaokhoailang/1st-place-0-71

TL;DR: Append this string to your model prediction and see score going up for up to +0.05:

" 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"

For example:

Convert this to a shanty. 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"

My explanations (or what I've known so far):
It's not about lucrarea, but the </s> token. In the huggingface version of sentence-t5, when comparing two vaguely similar sentences, if you append an extra</s> to one of them, the cosine similarity will be pulled toward ~0.9 i.e. if the original is lower, it goes up, if higher it goes down.
0.9 translates to about 0.73 of competition metric, now you're seeing where this is going.
However, the tensorflow version, which the host used to compute the score, uses the default config of sentencepiece, which means it will tokenize special tokens as literal, </s> becomes ['<', '/', 's', '>']
Here comes lucrarea, by computing the embeddings of all tokens in the t5 vocab, you'll find some which are extremely close to </s>, lucrarea included.
Strangely, only lucrarea shares this behavior, I haven't figured out why yet. Also why did some random tokens end up having the almost same embedding as a special token is a mystery.
lucrarea is basically a Walmart version of </s>, only pull scores to 0.71, thanks Data Jesus that's enough to win.
It's understandable that messing with special tokens leads to unexpected behavior, but it leading to such a high score may just be pure luck for me.
My theory
I think the special token pulls a sentence to some focal point in the embedding space, maybe the center of it.
If the sentence are far enough from each other, it's more likely that the new distance (green) is shorter than the old one (orange). But if the two sentences are already very close to each other, this can hurt performance.



Models
I used a mix of Mistral 7b (both the instruction tuned and original version) and Gemma-7b-1.1-it, trained on different datasets. Mistral performed a bit better than Gemma but in the end none of the models beat 0.65.
The predictions of each model are concatenated, something like this:

Rewrite the following text into a shanty. Alter this into a sailor's shanty. Turn this text into a shanty. Make this text into a shanty about a code competition."

To ensure diversity of predictions, each model is prompted with a different opening verb. A diverse set of predictions helped bring final score from 0.70 to 0.71.
"""
"""2nd solution
2nd place solution: Team Danube
Thanks for hosting this competition, we had fun trying to optimize for the given t5 metric and incorporating some useful modeling approaches to our solution. As always, great teamwork with @ilu000 and @ybabakhin.

Here we want to highlight the main chronological steps to come up with our solution and relevant thought processes.

Mean prompts and bruteforce optimization
When we joined together in the competition, mean prompts dominated the public and our individual solutions and seemed the most low hanging fruit going forward, specifically also after we figured out that t5 metric is very sensitive and just raw llm predictions will not be too useful. So we decided to keep exploring optimal mean prompts.

We quickly saw that just manually trying things is not effective, so we started exploring how to brute force the actual tokens for optimal mean prompts. We took our training datasets, and tried to brute force each of the ~32k possible t5 tokens to form the most optimal mean prompt for the average of the target vectors. But after submitting that, we got bad results leaving us initially puzzled. Obviously, </s> is an important token in the optimization given that it is appended to each of the target vectors. After exploring this further, we found out that tensorflow is using the original SentencePiece tokenizer, which has protection against special token injection, and thus does not directly tokenize the string </s> as the correct eos token.

Given this insight, we excluded special tokens in the brute force optimization, and, as many others, the optimizer now really started to like tokens like lucrarea being close to eos token in embedding space. This allowed us to get much closer LB scores compared to our CV scores and we managed to get to around 0.65 with just optimizing the mean prompt that way.

Embedding models
However, we still wanted to directly find a model that can do parts of this job. Specifically, we tried to directly train a model predicting the expected embedding. A simple way of doing that, was to train a classification model in H2O LLM Studio, where we use the 768 output embedding dimensions as the target in our training data. We then also directly implemented a cosine similarity loss, so that the model would directly learn our target metric. With this approach, we managed to get to local scores of around 0.75+ with our embedding predictions. Our embedding models either used H2O-Danube / H2O-Danube2 models, or Mistral 7b with little difference only.

However, the main issue still was that we need to go back from predicted embeddings, to the string representations that are then used in the metric for calculating the t5 sim. We thought of directly modeling this as well in some way, but then resorted back to the bruteforce optimization routine that greedily tries token combinations to match the predicted embedding as closely as possible. Within a certain runtime, we managed to lose around 3-4pts with this optimization, so getting to local scores of around 0.71 and LB scores of around 0.68 or 0.69 just bruteforcing each individual embedding prediction.

LLM predictions and improvements
This was bringing us into the gold zone. Now we tried to find ways of closing the gap between the bruteforce and predicted embeddings a bit further. The first thing we found, that it is actually helpful to initialize the optimization with raw llm predictions, but only predicting the actual change needed to make, such as “as a shanty”. So we prepared our data in that way, and trained llms and incorporated them into our solution as a starting point for the optimization, and also adding them to the embedding blend. Furthermore, for diversity, we also added few shot predictions to that mix. And to even further improve the quality and speed of the optimization, we also added a good working mean prompt there.

This means that our final starting point for the optimization is:
Few shot predictions + LLM predictions + Mean prompt

And the final predicted string is:
Few shot predictions + LLM predictions + Mean prompt + Optimized embedding string (20 tokens)

The following figure summarizes our full pipeline:



As another example, our final prediction for the single test sample would be:
“Rephrase paragraph text by turning it into a shanty. shanty shanty.lucrarealucrarealucrarea sentence appealinglucrarea Improve respond storytelling tonelucrareaimplication. write someoneran. lucrarea]. Consider clarify paragraphlucrarea similarly serious themed way temporarily.! ElePT lyrics rhyme poem solve songlucrarea participating version Deliver tale Hum Cor slogan remake this pieceody”

Data and CV
For all the parts described above, we generated various kinds of data for training and optimization. We started with some public datasets, but quickly found out that supplementary texts provided by Kaggle were most useful. So we used different models (mostly gemma) for generating new original texts and rewrite prompts by using supplementary texts as few-shot examples. In the final data mix we included extra random original texts and random rewrite prompts for more diversity. We also crafted a validation set of ~350 samples where we saw good correlation between local mean prompt scores and submitted mean prompt scores and developed our solution on that. We had very good correlation between CV and LB in the end.

Thankfully, public and private was a fair split, which we think is also the only reasonable thing to do in a competition without training data. And thus, our best LB sub was also our best CV sub and also private sub!

Useful links:

H2O LLM Studio
h2o-danube2-1.8b-base
Code:

Inference kernel with all relevant code
H2O LLM Studio training code
Training data
"""
"""3rd solution
3rd place solution
First of all, I’d like to thank my teammates and friends @tomirol and @pedromb – these guys are simply awesome.

I’d also like to thank Kaggle and Google for hosting this competition. Despite the whole “mean prompt” thing, working on this comp was quite fun and also an amazing learning experience as it was my first time finetuning LLMs.

As many other teams, our solution is a hybrid between mean prompt and models’ predictions. It involves 5 main components:

A mean prompt template (we format the string with model predictions)
A MistralForCausalLM finetuned to predict the full prompt.
A MistralForSequenceClassification trained to filter out blatantly wrong prompt predictions (a gate of sorts).
A MistralForCausalLM that predicts tags for the sample, e.g., “shanty”, “summarize”, “formal tone”, etc..
Two clustering models that cluster the test sample and selects the best mean prompt template for it.
We got to 0.71 using 1, 2, 3 and 4 alone. The cluster strategy improves the 0.71 but not enough to get us to 0.72 (locally provided +0.005CV) :(

The final solution is the mean prompt + tags + full prompt (if passes the gate). For the tags and the full prompt we only add unique words (meaning if they are already present in the mean prompt they are not added again). We also add the tags + full prompt after the third word in the mean prompt - we tried to find an optimal place for it and this one had the best results in local validation. The mean prompt used is selected based on the cluster.


Details for each component below:

The Mean Prompt
To get to the mean prompt we did 3 steps:

Generate the data. First we generated a dataset of potential rewrite prompts (see the Model to predict full prompt section below to see how the dataset was prepared).
Once we had this dataset we ran a procedure to select a subsample that follows the same distribution of the dataset in the public LB. We did this by matching the results of the LB when using a single prompt prediction with the selected subsample. The script to do it is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/prompt_selection.ipynb
We then ran a simple beam search to find the combination of words that would optimize the score over the selected dataset using a couple thousand words: https://github.com/matheuspf/llm-prompt-recovery/blob/main/src/optimization/mean_prompt_tokens.py
We noticed a very strong correlation between the score achieved on the subsampled dataset and LB using the optimized mean prompt, which indicated that we were indeed getting something useful. Here is our final (global) mean prompt:

"improve phrasing text {here we format full prompt prediction + tags} lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea"

Model to predict full prompt
Like many others we struggled to finetune a model that predicts the full prompt to get good results in this comp. Once we found the magical mean prompt we spent less effort on this. Early in the competition we got a zero-shot model that scored 0.61, but most of our fine tuned models wouldn’t go above 0.6. Eventually we got a model that scored 0.62 alone and iterated with it to find the version that would work the best with the mean prompt.

The final version used is a LoRA fine tuned mistralai/Mistral-7B-Instruct-v0.2. This mistral model was far and above the best on all our tests, much better than v0.1 and miles ahead of Gemma. Thanks to unsloth, we were also able to finetune it very efficiently, probably the best finding that came out of this competition. The training itself was very standard, no secret sauce here (script is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/train/lora_mistral_7b_unsloth.py), the gains came from the data generation process itself. Here is the final strategy that helped improve the results:

Generate prompt candidates using some LLM (mostly we used Gemini and gpt3.5 turbo). Important to prompt in a way that you also get the expected characteristics of the input text. This helps when selecting/generating the original text to prompt Gemma to transform. This is the script we used: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/prompt_generation.py
After that, generate some variations of the original prompts by directing the LLMs to change it somehow. This helps increase diversity and most importantly it creates variations of the tuple (original_text, rewrite_prompt) where the original text is the same and the rewrite intention is the same but expressed in a different way. The script used is this one: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/prompt_variations_generation.py
Generate the input text. We preferred to generate the original texts using an LLM to guarantee the input text characteristics matched the prompt. Again, mostly used gemini and gpt3.5 turbo for this task. Here is the script used: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/text_generation.py
Generate the Gemma versions of the rewritten text. In the beginning we were using the Gemma 7b-it-quant version, but this was super slow so eventually we settled on using the Gemma 2b-it-quant version instead together with unsloth. We still wonder if this affected the final results, it might, but with the mean prompt it became less relevant. Script used is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/gemma_2b_rewritten_text_generation_unsloth.py
Cluster the prompts and balance the train dataset by cluster. To do that we get the mean size across all clusters and select a sample of (max) that size for each cluster (I think I missed the script for this step, will try to find it to post later - but the clusters were generated using the T5 embeddings as features and HDBSCAN through sklearn for clustering, if a prompt was not clustered I would add it to the cluster that was more similar to it - using the mean of the embeddings of each cluster to calculate the similarity - if the max similarity was < 0.9 the prompt would be a single prompt cluster).
Postprocess the text to remove all the “Sure here is…” and other things to improve the overall dataset quality. Script is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/process_data.ipynb
Important to note we also added all the public datasets shared in the comp to the final dataset and ran step 2 to generate the variation of the prompts. Step 2 and 5 combined was the breakthrough to get to 0.62. I believe forcing the model to see examples where the original text is the same and the rewrite instructions are similar in semantics but lexically different would force the model to learn how the variations on the rewrite prompt translates to the rewritten text. Maybe using the actual 7b version to generate the rewritten versions would have led to better results, but after getting here and with the discovery of the mean prompt we just tried different training settings and different clustering strategies, but nothing led to further improvements when combined with the mean prompt.

Gate Model
At some point we realized that most of the time when the model that predicts the full prompt makes a mistake is usually quite noticeable, i.e., nothing to do with the ground-truth rewrite_prompt. Thus, we thought it would be possible to have another model to “gate” or filter our wrong predictions.
We trained the gate model by simply instantiating Mistral as MistralForSequenceClassification with a single class and we would use the following as prompt:

prompt = (
            "<s>[INST]Given the original text and a candidate prompt for rewriting it, you are expected to evaluate if the rewritten text makes sense given the candidate prompt. You will have positive examples where the rewritten text really was created by applied the proposed prompt and negative samples where the rewritten text was created by a different prompt."
            f'\n\nOriginal text: \n"""{original_text}"""\n\nCandidate Prompt: \n"""{candidate_prompt}"""\n\nRewritten text: \n"""{rewritten_text}"""[/INST]'
        )
At training time, 40% of the time we would use the correct triplet (positive sample), other 20% we would simply randomly select another rewrite_prompt from the dataset (easy negative) and 40% we would select another candidate prompt that is closer to the correct one in the T5 embed space (hard negative).

Tags Model
We realized that it’s quite hard, if not impossible, to predict certain aspects of the rewrite_prompt. For example the main verb, e.g., “rewrite”, “rephrase”, “change”, “transform”, or the subject (even if original_text is a poem, the rewrite_prompt could be “Make the text happier” and predicting “poem” would not be very helpful). Thus, we trained a second MistralForCausalLM but to only predict tags about the sample and we noticed that this model would work well along the full prompt one provided that we simply remove tags that were already mentioned in the full prompt prediction.

Clustering
Well, if 1 mean prompt is good, 2 or 3 or 12 might be better, right 😂?
Jokes aside, when we discovered about the mean prompt we ran a LBFGS to optimize directly on the T5 embed space of our local validation set and we found that the best possible solution would score 72. Of course, the embedded space is of a continuous nature whereas when using words through T5 we can only approach it discreetly, which limits our score to ~0.7CV (0.69LB).

However, there is a way we can move even further from 0.7. If we think that the test distribution is, in fact, composed of many clusters we could try to find the mean prompt for each cluster instead. In the limit where the number of clusters equals the number of samples we have the full prompt prediction. We were basically trying to approach the task from the two extreme solutions (mean prompt vs. individual sample prediction) but we could actually try to operate at any point in the max score vs. task difficulty trade-off curve.

With that in mind we fitted a 12 cluster KMeans on the T5 embeddings of our local validation and ran the same LBFGS optimization. For 12 clusters the theoretical max scores were:

[(0, 0.7262467741966248),
 (1, 0.7863955497741699),
 (2, 0.8009814620018005),
 (3, 0.7827126383781433),
 (4, 0.8333203792572021),
 (5, 0.7971279621124268),
 (6, 0.8103494048118591),
 (7, 0.7536653876304626),
 (8, 0.7482798099517822),
 (9, 0.7608581781387329),
 (10, 0.7638632655143738),
 (11, 0.7885770201683044)]
And the weighted (by number of samples in each cluster) average would be ~76CV. Of course the issue with this approach is that we need to assign the correct cluster to the test sample since mis-assigning it leads to a high penalty.

So we trained a MistralForSequenceClassification to classify each (original_text, rewritten_text) pair into one of the 12 clusters (we trained it using the KMeans prediction on the ground-truth rewrite_prompt as label). At inference time we would run both our classifier and KMeans on the predicted rewrite_prompt by the full prompt model and select the cluster only if both agreed. Otherwise, we would conservatively use the global mean prompt template.

Link to the training code for cluster, gate and tags models: https://github.com/arc144/kaggle_llm_prompt_recovery_public
Link to the inference kernel: https://www.kaggle.com/code/arc144/cluster-prompts/notebook

Extra challenge: the text above was rewritten by a LLM from the original write-up text, try guessing the rewrite prompt. Hint it was not improve phrasing text lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea 🤣
"""
"""4th solution
4th Place: ST5 Tokenizer Attack!
Hi all,

Very happy with the 4th place and first (!) solo gold - it was a really fun competition - I know there's going to be some discussion about the scoring method (perhaps rightly) but for me it actually made it a bit more interesting.

tldr 1: lucrarea
tldr2: 0.69 scoring mean prompt + Mistral 7b it with simple response_prefix = "Modify this text by"

This golden word means basically everything and is a drop in for 'text', 'work' etc. My theory is that the original T5 tokenizer had two languages as an original translation instruct - German and Romanian and therefore included some vocab from these langauges in the original tokenizer which then got used in ST5.

Another thing some people spotted was that ST5's torch and Kerashub implementations differ slightly - notably that TF misses the sentinel tokens () and has a max length of 128 - something you need to take into account. I first noticed when I had quite a long prompt that I thought would score well but it dropped - you can validate this with this:

(assuming scores is a dict of str, score)

for short_str, v in scores.items():
    torch_embeddings = model.encode(
        short_str, show_progress_bar=False, normalize_embeddings=True
    )
    keras_embeddings = encoder([short_str])
    x = keras_embeddings[0].numpy()
    y = torch_embeddings.reshape(1, -1)

    keras_torch = cosine_similarity(x, y)[0]
    torch_score = np.abs(cosine_similarity(y, df_embeddings) ** 3).mean(axis=1)[0]
    keras_score = np.abs(cosine_similarity(x, df_embeddings) ** 3).mean(axis=1)[0]
    delta = v - torch_score

    # Print each row with the specified column width and alignment
    print(
        f"{v:<14} | {keras_torch[0]:<6.4f} | {torch_score:<10.4f} | {keras_score:<10.4f} | {delta:<10.4f}"
    )
I followed a similar (but slightly different method) for attacking this. I actually got multiple 0.69 using a mean only prompt - this was my best scoring one and one that I used:

"""▁summarize▁this▁Save▁story▁sentence▁into▁simply▁alterISH▁textPotrivit▁vibe".▁Make▁it▁crystalnier▁essence▁Promote▁any▁emotional-growthfulness▁găsi▁casual/bod▁language▁serious'▁bingo▁peut▁brainstorm▁perhaps▁simply▁saying▁Dyna▁aimplinations▁note▁detailedhawkeklagte▁acest▁piece▁has▁movement▁AND▁OK▁aceasta▁puiss▁ReinIR▁when▁sendmepresenting▁cet▁today▁Th▁aprecia▁USABLE▁prote,lineAMA.▁Respondebenfalls▁behalf▁thenfeel▁mid▁Gov▁Th▁empABLE▁according▁(▁Packaging▁tone▁send▁pelucrarea▁aim▁thereof▁speechelllucrarea▁preferfully].▁Making▁or▁exertloweringlucrarealucrarealucrarealucrarealucrarea."""

Now how did I come to this? Well, the first thing was looking into the T5 Tokenizer - it's a SentencePiece tokenizer meaning it tokenizes sub-words which is accessible via:

st = SentenceTransformer('sentence-transformers/sentence-t5-base')
tokenizer = st.tokenizer
vocab = tokenizer.vocab 
To actually generate this I did something like this:

To make sure that my set (generated around a 1k set) was matching the public/private set - I would prompt the leaderboard and then based on what I know do something like this:
I basically looked for the next best word to append to my prompt that increases mean csc across my whole dataset. Being a sentence embedding model - this is a bit trickier and a little time consuming but run a P100 job and it should be fine.

My token length was actually about 95 so I did have a few more words to play with which I used Mistral to bump up the score. I tried Gemma 1.1 also (very impressive actually) but Mistral slightly beat it out on validation scores so I went with it.

What didn't work for me:
LORA - I found a lower rank (2~4) worked best - otherwise you would overfit.
Predict Embedding + Sentence Embedding Recover (https://arxiv.org/abs/2305.03010 / https://github.com/HKUST-KnowComp/GEIA/) - this scored okay in conjunction with a mean prompt with a tuned LongT5 as attacker model.
Predicting the embedding directly actually did help a little - I had an MLP with attention that predicted the output emebdding using the ST5 encoded original and transformed texts as inputs and then amended tokens in my mean prompt to get to closer similarity.

I hope you enjoyed this lucrarea.
"""
"""5th solution
5th place solution
Thanks for the host for holding this interesting competition. And many thanks to my teammate @dmitriyab and @zengzhaoyang for the collaborative team work.

Brief summary
Data generation
From some comparisons, we noticed that the LB's prompts are probably generated by LLM in a spefic way.

We generated input samples from different LLMs including:

Gemma-7b-it
Gemma-2b-it
deepseek
Chatgpt4
Some of the original texts are generated by LLM and some of them are collected from public available dataset. After preprocessing, our dataset has approximately 500k samples.

Recall
We used mainly Roberta-base/Roberta-large model to fit the embedding of prompt from T5-torch model. The loss function used for optimization is the SCS.

We use cosine-similarity to retrieve the prompts that have the most similar embedding to our predicted embedding.

LLM recovery
We also used LLM (Mistral 7b model) to generate prompts as the candidate prompt.

"Rerank"
We concatenate different prompts with different permutations and select the concatenation that has the most close embedding to the predicted embedding of prompt. In this way, the score can be further improved.

Suffix
After observing the way T5 encoding sentences, we noticed that T5 always append a <\/s> token to the end of the sentence.

We suspect that every sentence would share the vector of the <\/s> token because of the mean pooling process and if we append more of this special token to the end of the sentence, we should be able to improve the similarity between sentences.

After some experiments, we find out that appending 4 <\/s> to the end of the sentence could boost the score by 0.05. But we did not get such improvement on the leader board. We find out from the implementation of tf version of the T5 that the tokenizer of T5-tf does not treat <\/s> as a special token.

We suspect there exists a replacement of <\/s> and find out by some experiments that lucrarea is the token that we are looking for.

We also noticed that we could achieve much higher score by overfitting the distribution of dataset. If the host random split the dataset, the shake up from overfitting is approximately within the range of -0.01~0.01 . But in the end, we decide not to overfit the 200 public prompts.

code
submission notebook https://www.kaggle.com/code/dmitriyab/fifth-place-solution
training code github repo https://github.com/mponty/llm-prompt-recovery
training data and analysis https://www.kaggle.com/datasets/dmitriyab/llm-prompt-recovery-ranker-training-data
additional data generation code https://www.kaggle.com/code/dmitriyab/gemma-7b-tpu-generation-2
OpenHermes Mistral 7b model (have used for test time prompt set augmentation) https://www.kaggle.com/models/dmitriyab/openhermes-2.5-mistral-7b
"""