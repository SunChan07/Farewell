import json
import re
import urllib.request

def download_and_filter_dataset(output_txt_path="romantic_dialogues.txt", max_words=15):
    url = "https://hf-mirror.com/datasets/vetertann/promease_chat/resolve/main/qna_train.jsonl.txt"
    local_file = "qna_train.jsonl.txt"

    print("⏳ Скачиваю файл напрямую через зеркало в обход блокировок...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response, open(local_file, 'wb') as out_file:
            out_file.write(response.read())
        print("✅ Файл успешно скачан и сохранен в папку проекта!")
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        return

    cleaned_dialogues = []
    russian_only = re.compile(r'^[а-яА-ЯёЁ\s.,!?—:]+$')

    with open(local_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                text_block = item.get("text", "")
            except:
                continue

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

    print(f"🎉 Сборка завершена! Создан чистый файл '{output_txt_path}'.")
    print(f"Собрано {len(cleaned_dialogues) // 2} идеальных диалогов.")

if __name__ == '__main__':
    download_and_filter_dataset()
