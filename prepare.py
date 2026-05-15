"""
Подготовка чистых данных
"""

import json
from collections import Counter
from ner_utils import tokenize_text, create_bio_tags, contains_russian_words
from filter_data import filter_dataset
from nltk.tokenize import TreebankWordTokenizer
import nltk
nltk.download('punkt', quiet=True)

def prepare_ner_data():
    print("Подготовка данных")
    print("="*60)

    # Фильтруем данные
    examples, stats = filter_dataset(
        csv_path="classification_new.csv",
    )

    # Преобразуем в NER-формат
    ner_examples = []
    nltk_tokenizer = TreebankWordTokenizer()

    for ex in examples:
        # 1. Получаем спаны токенов через NLTK
        token_spans = list(nltk_tokenizer.span_tokenize(ex['text']))
        
        # 2. Размечаем теги
        tags = []
        for token_start, token_end in token_spans:
            if token_start >= ex['start'] and token_end <= ex['end']:
                tags.append("B-SUPPLIER" if len([t for t in tags if t != 'O']) == 0 else "I-SUPPLIER")
            else:
                tags.append("O")
        
        # 3. Собираем токены
        tokens = [ex['text'][s:e] for s, e in token_spans]
        
        ner_examples.append({
            "tokens": tokens,
            "ner_tags": tags
        })

    print(f"Подготовлено NER-примеров: {len(ner_examples)}")

    if len(ner_examples) == 0:
        print("НЕТ ПРИМЕРОВ ДЛЯ ОБУЧЕНИЯ!")
        return None

    # Статистика
    all_tags = []
    for ex in ner_examples:
        all_tags.extend(ex['ner_tags'])

    tag_counts = Counter(all_tags)
    total = len(all_tags)

    print("Распределение тегов:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag:10} - {count:5d} ({count/total*100:.1f}%)")

    # Сохраняем
    with open('clean_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(ner_examples, f, ensure_ascii=False, indent=2)

    print(f"Сохранено в clean_dataset.json")
    return ner_examples

if __name__ == "__main__":
    prepare_ner_data()