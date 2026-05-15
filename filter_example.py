"""
Фильтрация шума и противоречий
"""

import pandas as pd
import re
from typing import Tuple, Optional, Dict
from ner_utils import find_counterparty_in_text, contains_russian_words

def is_high_quality_example(
    text: str,
    counterparty_name: str,
) -> Tuple[bool, Optional[Dict]]:
    """
    Проверяет, является ли пример качественным для обучения.
    
    Критерии отсева:
        - название не найдено в тексте
        - название короче 3 символов
        - в тексте нет русских слов
        - LLM вернула невалидное значение («участник №...»)
    
    Возвращает:
        (True, info) если пример качественный,
        (False, None) если пример нужно отбросить
    """
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
