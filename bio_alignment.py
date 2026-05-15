"""
Выравнивание токенов и BIO-разметка
"""

import nltk
from nltk.tokenize import TreebankWordTokenizer

nltk.download('punkt', quiet=True)

def tokenize_with_spans_nltk(text: str) -> list:
    """Токенизация с позициями в исходном тексте"""
    tokenizer = TreebankWordTokenizer()
    spans = list(tokenizer.span_tokenize(text))
    tokens = []
    for start, end in spans:
        tokens.append({
            'token': text[start:end],
            'start': start,
            'end': end
        })
    return tokens

def align_to_bio(tokens: list, entity_start: int, entity_end: int, entity_type: str = 'SUPPLIER') -> list:
    """
    Выравнивание позиций сущности на токены и создание BIO-тегов
    """
    tags = []
    entity_indices = []
    
    current_pos = 0
    for idx, token in enumerate(tokens):
        token_start = current_pos
        token_end = current_pos + len(token)
        
        if not (token_end <= entity_start or token_start >= entity_end):
            entity_indices.append(idx)
        
        current_pos = token_end + 1
    
    for i, idx in enumerate(range(len(tokens))):
        if idx in entity_indices:
            if entity_indices.index(idx) == 0:
                tags.append(f'B-{entity_type}')
            else:
                tags.append(f'I-{entity_type}')
        else:
            tags.append('O')
    
    return tags
