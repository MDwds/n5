from datasets import load_dataset, DatasetDict, Dataset
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np


import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# PEFT (Parameter-Efficient Fine-Tuning)
# LoRA (Low rank adaptation) 是现在最常用的 PEFT 方法。





# dataset = load_dataset("glue", "sst2",data_files=data_dict)
dataset = load_dataset("glue", "sst2")
print("dataset:", dataset)

positive_rate = np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])
print("positive_rate:", positive_rate)

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

model_checkpoint = 'roberta-base'

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

print("model:", model)

# 创建 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# 如果tokenizer没有pad_token，设置tokenizer的pad_token为`[pad]`
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # 调整模型词向量层的大小,使其匹配tokenizer的词汇表大小。
    # 当我们自定义添加了特殊词汇(如本例中的pad_token)到tokenizer后,这些词还没有对应的词向量。
    # 为了让模型可以正确处理这些词汇,需要扩大模型原始的词向量层,使其大小匹配tokenizer词汇表的大小。
    # resize_token_embeddings就是用来调整词向量层大小的方法。len(tokenizer)可以获取tokenizer的词汇表大小。
    model.resize_token_embeddings(len(tokenizer))

# 创建 tokenize function
def tokenize_function(examples):
    # 提取文本
    text = examples["sentence"]

    # 从左边截取，最大长度512
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 创建 data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load accuracy evaluation metric
accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

# 定义例子
text_list = ["a feel-good picture in the best sense of the term .",  # positive
             "resourceful and ingenious entertainment .",   # positive
             "it 's just incredibly dull .",   # negative
             "the movie 's biggest offense is its complete and utter lack of tension .",  # negative
             "impresses you with its open-endedness and surprises .",   # positive
             "unless you are in dire need of a diesel fix , there is no real reason to see it ."  # negative
            ]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # 计算 logits
    logits = model(inputs).logits
    # 把 logits 转换为 label
    predictions = torch.argmax(logits)
    
    print(text + " - " + id2label[predictions.tolist()])


peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        #  参数大小看它，
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['query'])

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 16
# num_epochs = 5
num_epochs = 1

training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

model.to('cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])

