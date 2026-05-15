"""
Тестирование обученной модели
"""

import torch
import nltk
from nltk.tokenize import TreebankWordTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification

nltk.download('punkt', quiet=True)

def load_model(model_path="./supplier_ner_model"):
    """Загрузка модели"""
    try:
        print(f"Загружаю модель из {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        print("Модель загружена")
        return tokenizer, model
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None, None

def predict_ner(text, tokenizer, model):
    """Предсказание сущностей"""
    if tokenizer is None or model is None:
        return []
    
    nltk_tokenizer = TreebankWordTokenizer()
    token_spans = list(nltk_tokenizer.span_tokenize(text))
    tokens = [text[s:e] for s, e in token_spans]

    inputs = tokenizer(
        tokens,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        padding=True
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0]
    word_ids = inputs.word_ids()
    
    predicted_tags = []
    for i, (pred, word_id) in enumerate(zip(predictions, word_ids)):
        if word_id is not None:
            tag = model.config.id2label[pred.item()]
            predicted_tags.append((word_id, tokens[word_id], tag))
    
    entities = []
    current_entity = []
    current_type = None
    
    for word_id, token, tag in sorted(predicted_tags, key=lambda x: x[0]):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith('I-') and current_type == tag[2:]:
            current_entity.append(token)
        elif tag == 'O' and current_entity:
            entities.append((' '.join(current_entity), current_type))
            current_entity = []
            current_type = None
    
    if current_entity:
        entities.append((' '.join(current_entity), current_type))
    
    return entities

def main():
    print("ТЕСТИРОВАНИЕ МОДЕЛИ SUPPLIER NER")
    print("="*60)
    
    tokenizer, model = load_model()
    if tokenizer is None:
        return
    
    test_texts = [
        "Договор поставки с ООО ТехноСтрой",
        "Акт выполненных работ от ООО Альфа групп",
        "Акт сверки с ИП Коваленко А.А.",
        "Контракт с АО Транснефть",
    ]
    
    print("\nТЕСТОВЫЕ ПРИМЕРЫ:")
    
    for text in test_texts:
        print(f"\nТекст: '{text}'")
        entities = predict_ner(text, tokenizer, model)
        
        if entities:
            print("Найдены сущности:")
            for entity, etype in entities:
                print(f"   '{entity}' -> {etype}")
        else:
            print("Сущности не найдены")
    
    print("\nТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    main()