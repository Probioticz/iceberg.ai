# Model Training Guide

This guide will help you:
- Collect safe public conversational datasets
- Fine-tune an open-source model for your chatbot
- Use your trained model with the FastAPI backend

---

## 1. Choose a Base Model

- For small/demo projects: [`distilgpt2`](https://huggingface.co/distilgpt2), [`gpt2`](https://huggingface.co/gpt2)
- For medium/production & coding tasks: [Llama 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Phi-3](https://huggingface.co/microsoft/phi-3-mini-128k-instruct)
- For coding: [CodeLlama](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf), [StarCoder](https://huggingface.co/bigcode/starcoder2-15b)

---

## 2. Collect Safe Data

### General Chatbot:
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/)
- [The Pile](https://huggingface.co/datasets/EleutherAI/the_pile)
- [Wikipedia](https://huggingface.co/datasets/wikipedia)
- [ShareGPT](https://huggingface.co/datasets/ChillySalmon/ShareGPT_Vicuna_unfiltered)

### Kids/Teens/Edu-Safe:
- [BookCorpus](https://huggingface.co/datasets/bookcorpus)
- [Open Assistant Conversations](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Code Q&A](https://huggingface.co/datasets/synthtext/code_search_net)

### Coding Module:
- [CodeSearchNet](https://huggingface.co/datasets/code_search_net)
- [StackExchange (filtered)](https://huggingface.co/datasets/stackexchange)

#### Filter/Review all data for safety, appropriateness, and privacy (especially if adding new datasets!).

---

## 3. Sample Data Collection Script

```python
from datasets import load_dataset
with open("train.txt", "w", encoding="utf-8") as f:
    # Example: load Wikipedia for general language
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    for row in wiki:
        f.write(row["text"].replace("\n", " ") + "\n")
```

You may similarly collect code samples for kids & teens using CodeSearchNet or filtered StackExchange answers.

---

## 4. Fine-tune The Model

You can use popular HuggingFace scripts, or adapt the sample below for small models:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def load_dataset(file_path, tokenizer):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)

train_dataset = load_dataset("train.txt", tokenizer)
val_dataset = load_dataset("val.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=1000,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("./model_output")
```

---

## 5. Using Your Model

- Place your trained/fine-tuned model in `backend/model_output`
- The FastAPI backend will use it for responses
- Test the chatbot and coding modules in your UI!

---

## 6. Tips for Safe & Effective Fine-tuning

- Review final output generations for age suitability
- Consider prompt engineering during training (e.g., preprending "You are a friendly tutor for kids age 9…")
- Augment with custom filtered data if needed
- Always re-check for data leaks or sensitive information

---

## Need More Performance or Features?

- Use more powerful models: Llama 2/3, Mistral, Phi, StarCoder
- Host using HuggingFace Inference Endpoints, Replicate.com, or your own cloud GPU

---

**Happy Training! Want a data collection template or help choosing the model? Open an issue, pull request, or discussion in the repo!**
