def filter_content(text: str, age: int) -> str:
    filters = {
        "9": ["violence", "dating", "personal info", "curse"],
        "13": ["strong profanity", "self-harm", "drugs"],
        "16": ["extreme profanity", "explicit adult topics"]
    }
    age_key = "9" if age <= 9 else "13" if age <= 13 else "16"
    for word in filters[age_key]:
        text = text.replace(word, "***")
    return text
