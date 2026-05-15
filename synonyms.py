"""
—ловарь синонимов организационно-правовых форм
"""

SYNONYMS = {
    "ип": ['ип', 'индивидуальный предприниматель', 'физическое лицо'],
    "ооо": ['ооо', 'общество с ограниченной ответственностью'],
    "ао": ['ао', 'акционерное общество'],
}

def normalize_company_type(company_type: str) -> str:
    """
    ѕриводит организационно-правовую форму к каноническому виду.
    
    ѕараметры:
        company_type: исходна€ форма (например, "ќбщество с ограниченной ответственностью")
    
    ¬озвращает:
        нормализованна€ форма (например, "ооо")
    """
    company_type = company_type.lower().strip()
    for canonical, variants in SYNONYMS.items():
        if company_type in variants or company_type == canonical:
            return canonical
    return company_type
