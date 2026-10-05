import re
from datasets import load_dataset

def download_and_adapt_promease(output_txt_path="romantic_dialogues.txt", max_words=15):
    print("⏳ Загружаю датасет напрямую из Hugging Face...")
    dataset = load_dataset("vetertann/promease_chat")

    cleaned_dialogues = []
    russian_only = re.compile(r'^[а-яА-ЯёЁ\s.,!?—:]+$')

    for item in dataset["train"]:
        text_block = item.get("text", "")
        if "### Human:" not in text_block or "### Assistant:" not in text_block:
            continue

        parts = text_block.split("### Assistant:")
        human_part = parts[0].replace("### Human:", "").strip()
        assistant_part = parts[1].strip()

        human_sentences = re.split(r'(?<=[.!?])\s+', human_part)
        assistant_sentences = re.split(r'(?<=[.!?])\s+', assistant_part)

        human_sentence = re.sub(r'https?://\S+|www\.\S+|\d+', '', human_sentences[0])
        assistant_sentence = re.sub(r'https?://\S+|www\.\S+|\d+', '', assistant_sentences[0])

        if not russian_only.match(human_sentence) or not russian_only.match(assistant_sentence):
            continue

        user_words = human_sentence.split()
        bot_words = assistant_sentence.split()

        if len(user_words) > max_words or len(bot_words) > max_words:
            continue
        if len(user_words) < 2 or len(bot_words) < 2:
            continue

        cleaned_dialogues.append(f"Пользователь: {' '.join(user_words)}")
        cleaned_dialogues.append(f"Бот: {' '.join(bot_words)}")

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for line in cleaned_dialogues:
            f.write(line + "\n")

    print(f"🎉 Готово! Создан чистый файл '{output_txt_path}'. Собрано {len(cleaned_dialogues) // 2} идеальных коротких диалогов.")

if __name__ == '__main__':
    download_and_adapt_promease()
