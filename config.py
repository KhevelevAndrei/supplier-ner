"""
Конфигурация проекта
"""

# Пути к файлам
CSV_PATH = "classification_new.csv"
CLEAN_DATASET_PATH = "clean_dataset.json"
MODEL_PATH = "./supplier_ner_model"
LOGS_PATH = "./logs"

# Модель
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LENGTH = 128

# Обучение
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
EPOCHS = 20
WARMUP_RATIO = 0.1