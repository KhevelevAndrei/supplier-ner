"""
Обучение модели с одним типом сущности SUPPLIER
"""

import json
import numpy as np
import torch
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
import config

def train_model():
    print("Обучение модели SUPPLIER NER")
    print("="*60)

    # 1. Загружаем данные
    with open(config.CLEAN_DATASET_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"Загружено примеров: {len(raw_data)}")

    # 2. Подготовка меток
    all_tags = []
    for item in raw_data:
        all_tags.extend(item['ner_tags'])

    unique_tags = sorted(set(all_tags))
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    print(f"Уникальные теги: {unique_tags}")

    # 3. Создаём датасет
    dataset = Dataset.from_list(raw_data)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Разделение: {len(train_dataset)} train / {len(val_dataset)} val")

    # 4. Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(unique_tags),
        id2label=id2tag,
        label2id=tag2id,
        ignore_mismatched_sizes=True
    )

    # 5. Функция токенизации
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(tag2id.get(label[word_idx], -100))
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)

    # 6. Функция метрик
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = []
        true_labels = []
        
        for pred, label in zip(predictions, labels):
            pred_seq = [id2tag[p] for p, l in zip(pred, label) if l != -100]
            label_seq = [id2tag[l] for p, l in zip(pred, label) if l != -100]
            if pred_seq and label_seq:
                true_predictions.append(pred_seq)
                true_labels.append(label_seq)
        
        return {
            "precision": precision_score(true_labels, true_predictions, scheme=IOB2, zero_division=0),
            "recall": recall_score(true_labels, true_predictions, scheme=IOB2, zero_division=0),
            "f1": f1_score(true_labels, true_predictions, scheme=IOB2, zero_division=0),
        }

    # 7. Настройка обучения
    training_args = TrainingArguments(
        output_dir=config.MODEL_PATH,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=config.LOGS_PATH,
        logging_steps=10,
        report_to="none",
        seed=42,
        save_total_limit=2,
        warmup_ratio=config.WARMUP_RATIO
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 8. Обучение
    print("НАЧАЛО ОБУЧЕНИЯ")
    trainer.train()

    # 9. Финальные метрики
    print("ФИНАЛЬНЫЕ МЕТРИКИ НА ВАЛИДАЦИИ:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # 10. Сохраняем модель
    trainer.save_model(config.MODEL_PATH)
    tokenizer.save_pretrained(config.MODEL_PATH)
    print(f"Модель сохранена в {config.MODEL_PATH}")

if __name__ == "__main__":
    train_model()