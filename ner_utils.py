"""
Модуль утилит для NER с одним типом сущности SUPPLIER
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from fuzzysearch import find_near_matches

# Синонимы типов
SYNONYMS = {
    "ип": ['ип', 'индивидуальный предприниматель', 'физическое лицо'],
    "ооо": ['ооо', 'общество с ограниченной ответственностью'],
    "ао": ['ао', 'акционерное общество'],
}

def get_variations(company_name, company_type):
    """Генерирует варианты написания с типом"""
    if not isinstance(company_name, str):
        return [company_name]
    results = [company_name]
    company_type = str(company_type).lower()
    for syn in SYNONYMS.get(company_type, []):
        results.append(f"{syn} {company_name}")
        results.append(f"{company_name} {syn}")
    return results

def remove_overlapping_spans(spans):
    """Убирает пересекающиеся span'ы"""
    spans = sorted(spans, key=lambda x: (x[3], x[0], -(x[1] - x[0])))
    result = []
    last_end = -1
    for start, end, alias, dist in spans:
        if start >= last_end:
            result.append((start, end, alias))
            last_end = end
    return result

def find_counterparty_fuzzy(text, company_name, company_type, max_dist=2):
    """Поиск с учётом опечаток и вариаций"""
    if pd.isna(company_name):
        return []
    
    text = str(text).lower()
    company_name = str(company_name).lower()
    company_type = str(company_type).lower()
    
    variations = get_variations(company_name, company_type)
    spans = []
    
    for variant in variations:
        matches = find_near_matches(variant, text, max_l_dist=max_dist)
        spans.extend([(m.start, m.end, text[m.start:m.end], m.dist) for m in matches])
    
    return remove_overlapping_spans(spans)

def contains_russian_words(text: str, min_words: int = 2) -> bool:
    """Проверяет, есть ли в тексте русские слова"""
    words = re.findall(r'[а-яА-ЯёЁ]{3,}', text)
    return len(words) >= min_words

def find_counterparty_in_text(text: str, counterparty_name: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """Ищет контрагента в тексте"""
    if pd.isna(text) or pd.isna(counterparty_name):
        return False, None, None

    text_str = str(text)
    name_str = str(counterparty_name).strip()

    if not name_str or name_str.lower() == 'nan':
        return False, None, None

    # 1. Прямой поиск
    if name_str in text_str:
        start = text_str.find(name_str)
        return True, start, start + len(name_str)

    # 2. Поиск без кавычек
    clean_name = re.sub(r'[«»"\'`]', '', name_str).strip()
    if clean_name and clean_name in text_str:
        start = text_str.find(clean_name)
        return True, start, start + len(clean_name)

    # 3. Регистронезависимый поиск
    text_lower = text_str.lower()
    name_lower = name_str.lower()
    
    if name_lower in text_lower:
        start = text_lower.find(name_lower)
        return True, start, start + len(name_str)

    return False, None, None

def tokenize_text(text: str) -> List[str]:
    """Токенизация на слова и знаки препинания"""
    return re.findall(r'\b\w+\b|[^\w\s]', str(text))

def create_bio_tags(tokens: List[str], entity_indices: List[int], entity_type: str = 'SUPPLIER') -> List[str]:
    """BIO разметка"""
    ner_tags = ['O'] * len(tokens)

    if not entity_indices:
        return ner_tags

    entity_indices = sorted(entity_indices)

    for i, idx in enumerate(entity_indices):
        if i == 0:
            ner_tags[idx] = f'B-{entity_type}'
        else:
            ner_tags[idx] = f'I-{entity_type}'

    return ner_tags