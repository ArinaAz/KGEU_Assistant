import transformers
import os
from huggingface_hub import notebook_login

token = "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"
#без ввода токена вручную
os.environ["HUGGINGFACE_TOKEN"] = token

notebook_login()


#from huggingface_hub import notebook_login

#notebook_login()
from transformers.utils import send_example_telemetry

# Отправка телеметрии о примере
send_example_telemetry("translation_notebook", framework="pytorch")

model_checkpoint = "Helsinki-NLP/opus-mt-ru-en"

from datasets import load_dataset, load_metric

# Загрузка набора данных WMT16 для русско-английского перевода
raw_datasets = load_dataset("wmt16", "ru-en")
metric = load_metric("sacrebleu")

# Вывод информации о наборе данных
raw_datasets

# Вывод первого примера из тренировочного набора данных
raw_datasets["train"][0]

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

# Вывод нескольких случайных примеров из тренировочного набора данных
show_random_elements(raw_datasets["train"])

# Вывод метрики BLEU
metric

# Создание фиктивных предсказаний и меток для вычисления метрики BLEU
fake_preds = ["hello there", "general kenobi"]
fake_labels = [["hello there"], ["general kenobi"]]
metric.compute(predictions=fake_preds, references=fake_labels)

from transformers import AutoTokenizer

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if "mbart" in model_checkpoint:
    tokenizer.src_lang = "en-XX"
    tokenizer.tgt_lang = "ro-RO"

# Применение токенизатора к одному предложению
tokenizer("Hello, this one sentence!")

# Применение токенизатора к нескольким предложениям
with tokenizer.as_target_tokenizer():
    print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate Russian to En: "
else:
    prefix = ""

max_input_length = 128
max_target_length = 128
source_lang = "ru"
target_lang = "en"
# выполняет предварительную обработку данных для подготовки модели к обучению или применению.
def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]#формируется список inputs, который содержит предложения на исходном языке. 
    targets = [ex[target_lang] for ex in examples["translation"]]#формируется список targets, который содержит целевые предложения, представленные в examples["translation"]. Они выбираются по ключу target_lang.
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)#используется токенизатор (предположительно ранее инициализированный), чтобы преобразовать список inputs в численное представление, подходящее для подачи на вход модели. Токенизатор разбивает каждое предложение на токены и может выполнять дополнительные операции, такие как ограничение максимальной длины (max_length) и усечение (truncation), если это необходимо.

    # Настройка токенизатора для меток
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Подготовка данных для моделирования
preprocess_function(raw_datasets['train'][:2])

# Применение предобработки к набору данных
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Инициализация модели для машинного перевода
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
#создается объект args класса Seq2SeqTrainingArguments, который содержит различные аргументы и настройки для тренировки модели Seq2Seq 
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,  
    push_to_hub=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Замена -100 в метках, так как они не могут быть декодированы
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Некоторая обработка после декодирования
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Обучение модели
trainer.train()

#Данный код выполняет следующие действия:

#Импортируются необходимые модули и функции.
#Отправляется телеметрия о примере.
#Загружается предобученная модель машинного перевода из Hugging Face Model Hub.
#Загружается набор данных WMT16 для русско-английского перевода.
#Определяется метрика BLEU (sacreBLEU) для оценки качества перевода.
#Отображается информация о наборе данных.
#Отображается первый пример из тренировочного набора данных.
#Определяется функция для отображения случайных 