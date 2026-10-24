import re


def find_keywords(text: str, keywords: list[str]) -> list[dict]:
    """
    Находит ключевые слова в распознанном тексте.

    Поиск без учёта регистра.
    Считает полные совпадения слов, а не части других слов.
    """
    normalized_text = text.lower()
    found = []

    for keyword in keywords:
        normalized_keyword = keyword.strip().lower()

        if not normalized_keyword:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_keyword)
            + r"(?!\w)"
        )

        count = len(re.findall(pattern, normalized_text))

        if count > 0:
            found.append({
                "keyword": normalized_keyword,
                "count": count,
            })

    return found
