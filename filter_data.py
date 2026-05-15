"""
Модуль для отсеивания шумных данных
"""

import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from ner_utils import find_counterparty_in_text, contains_russian_words

def clean_text(text):
    """Нормализация текста"""
    if pd.isna(text):
        return text
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('​', '')
    return text

def is_high_quality_example(
    text: str,
    counterparty_name: str,
) -> Tuple[bool, Optional[Dict]]:
    """Проверяет качество примера"""
    if pd.isna(text) or pd.isna(counterparty_name):
        return False, None
    
    text_str = str(text)
    name_str = str(counterparty_name).strip()
    
    # Проверка на мусорные названия
    if re.match(r'участник\s+\d+', name_str.lower()):
        return False, None
    
    if len(name_str) < 3:
        return False, None
    
    if not re.search(r'[а-яА-Яa-zA-Z]', name_str):
        return False, None
    
    # Проверяем, есть ли в тексте русские слова
    if not contains_russian_words(text_str, min_words=1):
        return False, None
    
    # Ищем контрагента
    found, start, end = find_counterparty_in_text(text_str, name_str)
    
    if not found or start is None:
        return False, None
    
    return True, {
        "text": text_str,
        "counterparty_name": name_str,
        "start": start,
        "end": end,
        "found_text": text_str[start:end],
    }

def filter_dataset(
    csv_path: str,
    max_examples: Optional[int] = None
) -> Tuple[List[Dict], Dict]:
    """Отсеивает шум, оставляет качественные примеры"""
    df = pd.read_csv(csv_path)
    
    print(f"Загружено всего: {len(df)} строк")
    print(f"Из них с контрагентами: {df['counterparty_name'].notna().sum()}")
    
    # Фильтрация по class=1
    if 'class' in df.columns:
        df = df[df['class'] == 1]
        print(f"После фильтрации по class=1 осталось: {len(df)}")
    
    df['text'] = df['text'].apply(clean_text)
    
    quality_examples = []
    rejected = 0
    rejection_reasons = {
        "no_counterparty": 0,
        "participant_pattern": 0,
        "too_short": 0,
        "no_letters": 0,
        "no_russian_words": 0,
        "not_found_in_text": 0,
    }
    
    for idx, row in df.iterrows():
        if max_examples and len(quality_examples) >= max_examples:
            break
        
        if pd.isna(row.get('counterparty_name')):
            rejection_reasons["no_counterparty"] += 1
            rejected += 1
            continue
        
        is_quality, info = is_high_quality_example(
            row.get('text', ''),
            row.get('counterparty_name', ''),
        )
        
        if is_quality and info:
            quality_examples.append(info)
        else:
            rejected += 1
            name = str(row.get('counterparty_name', '')).lower()
            text = str(row.get('text', ''))
            
            if re.match(r'участник\s+\d+', name):
                rejection_reasons["participant_pattern"] += 1
            elif len(name) < 3:
                rejection_reasons["too_short"] += 1
            elif not re.search(r'[а-яА-Яa-zA-Z]', name):
                rejection_reasons["no_letters"] += 1
            elif not contains_russian_words(text):
                rejection_reasons["no_russian_words"] += 1
            else:
                rejection_reasons["not_found_in_text"] += 1
    
    stats = {
        "total": len(df),
        "accepted": len(quality_examples),
        "rejected": rejected,
        "acceptance_rate": len(quality_examples) / len(df) * 100 if len(df) > 0 else 0,
        "rejection_reasons": rejection_reasons
    }
    
    print(f"Принято: {len(quality_examples)} примеров ({stats['acceptance_rate']:.1f}%)")
    print(f"Отвергнуто: {rejected} примеров")
    
    return quality_examples, stats