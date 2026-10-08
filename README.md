# Assignment 11 – Fine-Tuning GPT-2 on a Custom Text Style

## 1. Objective

The objective of this assignment is to understand how fine-tuning helps a pretrained language model adapt to a specific writing style.

In this experiment, the small GPT-2 model was fine-tuned on a custom dataset containing approximately 200–300 lines of consistent technical dialogue.

The original GPT-2 model and the fine-tuned GPT-2 model were then given the same prompt, and their generated outputs were compared.

---

## 2. Model Used

**Model:** GPT-2 Small

**Model:** `gpt2`

**Library:** Hugging Face Transformers

**Training Framework:** Hugging Face Trainer

**Platform:** Google Colab

**Programming Language:** Python

---

## 3. Dataset

A custom dialogue dataset was created containing approximately 200–300 lines of text.

The dataset follows a consistent conversational and educational style.

### Example

```text
Alex: What is artificial intelligence?

Sam: Artificial intelligence is the ability of machines to perform tasks that normally require human intelligence.

Alex: What is machine learning?

Sam: Machine learning is a method where computers learn patterns from data.
```

The dataset focuses mainly on:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Neural Networks
- Natural Language Processing
- Transformers
- GPT-2
- Fine-Tuning
- Training
- Inference
- Text Generation

---

## 4. Dataset Cleaning

Before training, the dataset was cleaned to remove unnecessary information.

The following preprocessing steps were performed:

- Removed empty lines
- Removed unnecessary spaces
- Removed tab characters
- Removed unwanted formatting
- Kept only meaningful dialogue text
- Maintained a consistent question-and-answer structure

The cleaned dataset was saved as:

```text
custom_dialogue_dataset.txt
```

---

## 5. Tools and Libraries

The following Python libraries were used:

```text
Python
PyTorch
Transformers
Datasets
Accelerate
Google Colab
```

Installation:

```python
!pip install -q transformers datasets accelerate
```

---

## 6. Tokenization

GPT-2 requires text to be converted into tokens before training.

The GPT-2 tokenizer from Hugging Face was used.

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
```

The maximum sequence length used was 128 tokens.

---

## 7. Fine-Tuning Configuration

The GPT-2 model was fine-tuned using Hugging Face `Trainer`.

Main training parameters:

| Parameter | Value |
|---|---:|
| Model | GPT-2 |
| Epochs | 3 |
| Batch Size | 2 |
| Gradient Accumulation | 2 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Maximum Sequence Length | 128 |
| Optimizer | Hugging Face Trainer |
| Hardware | Google Colab GPU/CPU |

---

## 8. Training

The GPT-2 model was trained using:

```python
trainer.train()
```

The fine-tuned model was saved in:

```text
gpt2-finetuned/
```

The saved model can be loaded using:

```python
fine_tuned_model = GPT2LMHeadModel.from_pretrained(
    "./gpt2-finetuned"
)
```

---

## 9. Text Generation

The same prompt was provided to both models.

The two models tested were:

### Original GPT-2

The pretrained GPT-2 model before fine-tuning.

### Fine-Tuned GPT-2

The GPT-2 model after training on the custom dialogue dataset.

The same generation settings were used for both models to make the comparison fair.

---

## 10. Before-and-After Comparison

### Original GPT-2

The original GPT-2 model generally produces more general-purpose text because it has not been specifically trained on the custom technical dialogue style.

Example:

```text
Artificial intelligence is a field that has been developing rapidly...
```

### Fine-Tuned GPT-2

The fine-tuned model is more likely to follow the style of the custom dataset.

Example:

```text
Artificial intelligence is a technology that allows machines to learn from data.
Alex: What is machine learning?
Sam: Machine learning allows computers to learn patterns from examples.
```

---

## 11. Observed Differences

### Difference 1 – Writing Style

The original GPT-2 produces more general text.

The fine-tuned GPT-2 shows a stronger tendency toward the conversational question-and-answer format used in the training dataset.

---

### Difference 2 – Vocabulary

The original GPT-2 uses general vocabulary.

The fine-tuned model is more likely to use technical terms such as:

- Artificial Intelligence
- Machine Learning
- Neural Networks
- Transformer
- Dataset
- Tokenizer
- Training
- Fine-Tuning
- Inference

---

### Difference 3 – Conversation Structure

The fine-tuned model shows a stronger tendency to generate dialogue-like responses because the training dataset repeatedly contains conversations between Alex and Sam.

---

### Difference 4 – Repetition

The fine-tuned model may sometimes repeat phrases or patterns from the training dataset.

This can happen because the custom dataset is relatively small.

---

## 12. Results

The experiment demonstrates that fine-tuning can influence the behavior and writing style of a pretrained language model.

The fine-tuned GPT-2 model became more aligned with the style and vocabulary of the custom technical dialogue dataset.

The original GPT-2 model remained more general-purpose.

---

## 13. Conclusion

This experiment demonstrated how a pretrained GPT-2 language model can be adapted to a specific writing style using fine-tuning.

A custom dataset containing approximately 200–300 lines of technical dialogue was created and cleaned before training. The Hugging Face Transformers library and Trainer API were used to fine-tune GPT-2.

The same prompt was given to both the original and fine-tuned models. The comparison showed differences in vocabulary, conversation structure, writing style, and consistency.

Therefore, the experiment successfully demonstrates that fine-tuning can adapt a pretrained language model to a custom text style.

---

## 14. Files Included

```text
Assignment-11/
│
├── README.md
│
├── custom_dialogue_dataset.txt
│
├── base_model_output.txt
│
├── fine_tuned_model_output.txt
│
├── Assignment_11_GPT2_FineTuning.ipynb
│
└── gpt2-finetuned/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

---

## 15. Requirements

Install the required libraries using:

```bash
pip install transformers datasets accelerate torch
```

---

## 16. How to Run

1. Open the notebook in Google Colab.
2. Enable GPU from:

```text
Runtime → Change runtime type → T4 GPU
```

3. Install the required libraries.
4. Create/load the custom dataset.
5. Clean and tokenize the dataset.
6. Load GPT-2.
7. Configure the Hugging Face Trainer.
8. Start fine-tuning.
9. Save the fine-tuned model.
10. Generate text using the original GPT-2.
11. Generate text using the fine-tuned GPT-2.
12. Compare both outputs.

---

## 17. Learning Outcome

After completing this assignment, the following concepts were understood:

- Pretrained language models
- GPT-2 architecture
- Tokenization
- Fine-tuning
- Custom text datasets
- Hugging Face Transformers
- Hugging Face Trainer
- Text generation
- Temperature and sampling
- Model comparison
- Style adaptation
- Limitations of small datasets

---

## 18. Final Statement

Fine-tuning does not train a language model from scratch. Instead, it starts with an already pretrained model and further trains it on a specific dataset. This allows the model to become more aligned with the vocabulary, structure, and style represented in the custom training data.
