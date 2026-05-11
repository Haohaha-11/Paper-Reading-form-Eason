[← 返回 README](../README.md)

# A. Prompt Formats for Unsupervised Sample-Specific TTA

This section specifies the exact prompt templates used in our experiments. For each dataset, we construct a single text prompt $x$ and a gold answer $y$ . During unsupervised sample-specific TTA, the model adapts only on the prompt tokens (teacher-forced next-token NLL on $x$ ), and is then evaluated on the answer tokens (NLL on $y$ given $x$ ). For generation metrics (e.g., ROUGE-Lsum), we decode from the end-of-prompt cue (e.g., “Summary:”, “Answer:”, or “### Response:”) and compare the generated output to the gold answer.

> 💡 **appendix 总批注**: prompt format 是这篇能否成立的实验细节。因为 TTA loss 只看 prompt tokens，prompt 必须包含完整任务上下文和明确生成 cue；否则模型会优化一段和 answer 无关的文本，input perplexity 就更难转化为 answer quality。

# A.1. XSum (single-sentence news summarization)

XSum pairs a news article with a single-sentence summary. We use the following template, ending with “Summary:”.

# Prompt (XSum):

Summarize the following news article in one concise sentence.

Article: {document}

Summary:

The gold target is $y = \{ \mathsf { s u m m a r y } \}$ .

> 💡 **XSum 批注**: XSum 的 document 在 prompt 内，answer 是 summary；prompt-only TTA 可以适应文章主题、实体和风格，所以 Table 2 中 XSum ROUGE 提升最清楚。

# A.2. SQuAD (extractive reading comprehension)

SQuAD provides a passage and a question, and the answer is an exact text span from the passage. We explicitly instruct the model to output only the span and end the prompt with “Answer:”.

# Prompt (SQuAD):

You are given a passage and a question.
Answer with the exact text span from the passage. Output ONLY the answer span.

Passage: {context}

Question: {question}

Answer:

The gold target is the first annotated answer span, $y = \{ \mathsf { a n s w e r s } [ 0 ] \}$ (examples with empty answer lists are dropped).

> 💡 **SQuAD 批注**: SQuAD 的答案是 passage span，几乎完全在 prompt 中。动态 LoRA update 更像把 passage/question 的局部统计临时写进模型，因此 Qwen4B-Instruct layer-wise 5 steps 到 0.8306 的 ROUGE 提升比较符合预期。

# A.3. NQ-Open (open-domain short-answer QA)

NQ-Open contains a question and a short answer that may not appear in any given context. We use a minimal template tha requires outputting only the answer string:

# Prompt (NQ-Open):

Give a short answer of the following question. Output only the answer.

Question: {question} Answer:

> 💡 **NQ-Open 批注**: 这里 prompt 只有问题，没有 supporting context。优化 question 的 input perplexity 未必提供 answer 所需知识，解释了 Table 2 中 NQ-Open 收益小且方向不稳定：Qwen4B 变好，但 Llama3B-Instruct layer-wise 低于 No TTA。

# A.4. AdaptEval (instruction / domain / reasoning benchmarks)

AdaptEval includes multiple instruction- and QA-style datasets. We convert each example into a unified prompt/target pair. All AdaptEval prompts end with “### Response:” and the target is the reference response string.

Agriculture-QA (agriculture-qa). Each example contains a question and an answer string:

# Prompt (agriculture-qa):

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: {question}

### Response:

The gold target is $y = \{ { \mathrm { a n s w e r s } } \}$

Reasoning sets with instruction/output (MetaMathQA, gsm8k, logiqa). We use instruction as the instruction text and output as the target:

# Prompt (MetaMathQA / gsm8k / logiqa):

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: {instruction}

### Response:

The gold target is $y = \{ \mathtt { o u t p u t } \}$ .

General instruction-following schema with optional input (other AdaptEval subsets). For the remaining subsets, examples contain instruction, optional input, and output. If input is non-empty, we include an “### Input:” block:

# Prompt (with input):

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: {instruction}

### Input: {input}

### Response:

# Prompt (no input):

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: {instruction}

### Response:

In both cases, the gold target is $y = \{ \mathtt { o u t p u t } \}$ .

> 💡 **AdaptEval 批注**: AdaptEval 混合 domain、instruction、reasoning。prompt-only TTA 对 domain wording 可能有帮助，但对 GSM8K/MetaMath/LogiQA 这类推理输出，降低 instruction perplexity 不一定等价于更好的 reasoning trace 或 final answer。

You can have as much text here as you want. The main body must be at most 8 pages long. For the final version, one more page can be added. If you want, you can use an appendix like this one.

The \onecolumn command above can be kept in place if you prefer a one-column appendix, or can be removed if you prefer a two-column appendix. Apart from this possible change, the style (font size, spacing, margins, page numbering, etc.) should be kept the same as the main body.

> 💡 **抽取噪声批注**: 最后两段是模板残留，不影响方法。批读时真正需要回看的 appendix 内容是各数据集 prompt 是否给了足够上下文，因为这直接决定 unsupervised sample-specific TTA 的 proxy loss 是否合理。
