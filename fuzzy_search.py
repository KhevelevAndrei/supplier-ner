"""
Функция нечёткого поиска (fuzzysearch)
"""

import pandas as pd
from fuzzysearch import find_near_matches
from ner_utils import get_variations, remove_overlapping_spans

def find_counterparty_fuzzy(text, company_name, company_type, max_dist=2):
    """
    Ищет наименование поставщика в тексте с учётом опечаток и перестановок.
    
    Параметры:
        text: исходный текст документа
        company_name: наименование поставщика от LLM
        company_type: тип организации (ООО, ИП, АО)
        max_dist: максимальное расстояние Левенштейна (по умолчанию 2)
    
    Возвращает:
        список найденных совпадений (start, end, matched_text)
    """
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