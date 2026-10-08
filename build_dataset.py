import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    VOCAB_SIZE = 32000
    OUTPUT_DATA_FILE = "mixed_dialogues.txt"
    OUTPUT_VOCAB_FILE = "vocabulary.txt"
    
    # Лимиты символов под SRC_LEN=128 и TGT_LEN=128
    MAX_CHARS_LIMIT = 750   

    # Глобальный баланс 40% Ума / 60% Разговора (Суммарно 150 000 пар)
    TOTAL_PAIRS_TARGET = 150000
    INSTRUCT_LIMIT = int(TOTAL_PAIRS_TARGET * 0.40)  # 60 000
    DIALOGUE_LIMIT = int(TOTAL_PAIRS_TARGET * 0.60)  # 90 000

    print("🚀 Шаг 1: Загрузка датасетов из локального кэша Parquet...")
    instruct_ds = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split="train")
    chat_ds = load_dataset("Den4ikAI/russian_dialogues", split="train")

    print(f"\n🔮 Шаг 2: Фильтрация по сетке баланса под пословный токенизатор...")
    mixed_pairs = []
    instruct_count = 0
    chat_count = 0

    # 1. Сборка интеллекта [УМ]
    for item in tqdm(instruct_ds, desc="Сборка интеллекта [УМ]"):
        if instruct_count >= INSTRUCT_LIMIT:
            break
        q, a = "", ""
        
        string_values = [str(v).strip() for v in item.values() if v is not None and len(str(v).strip()) > 0]
        if not string_values:
            for v in item.values():
                if isinstance(v, list):
                    for sub_v in v:
                        if isinstance(sub_v, dict):
                            string_values.extend([str(val).strip() for val in sub_v.values() if val])
        
        banned_words = {"user", "bot", "gpt", "human", "system"}
        valid_strings = [s for s in string_values if s.lower() not in banned_words and len(s) > 3]
        
        if len(valid_strings) >= 2:
            q = valid_strings[0]
            a = valid_strings[1]

        if not q or not a:
            continue
            
        if len(q) <= MAX_CHARS_LIMIT and len(a) <= MAX_CHARS_LIMIT:
            # Важно: кодируем двоеточия так, как прописано в вашем load_vocab!
            q_clean = q.replace("\n", " ").replace("\t", " ").replace(":", "<COLON>")
            a_clean = a.replace("\n", " ").replace("\t", " ").replace(":", "<COLON>")
            mixed_pairs.append((q_clean, a_clean))
            instruct_count += 1

    # 2. Сборка повседневной речи [РАЗГОВОР]
    for item in tqdm(chat_ds, desc="Сборка разговорной речи [РАЗГОВОР]"):
        if chat_count >= DIALOGUE_LIMIT:
            break
        if "question" in item and "answer" in item:
            q = str(item["question"]).strip()
            a = str(item["answer"]).strip()
            
            if q and a and len(q) <= MAX_CHARS_LIMIT and len(a) <= MAX_CHARS_LIMIT:
                q_clean = q.replace("\n", " ").replace("\t", " ").replace(":", "<COLON>")
                a_clean = a.replace("\n", " ").replace("\t", " ").replace(":", "<COLON>")
                mixed_pairs.append((q_clean, a_clean))
                chat_count += 1

    print(f"\n💾 Запись сбалансированного микса в файл '{OUTPUT_DATA_FILE}'...")
    with open(OUTPUT_DATA_FILE, "w", encoding="utf-8") as f:
        for q, a in mixed_pairs:
            f.write(f"{q}\t{a}\n")

    print(f"\n🧠 Шаг 3: Генерация базового стартового словаря для DynamicTokenizer...")
    # Записываем строго спецтокены в формате слово:индекс, который требует ваш load_vocab
    with open(OUTPUT_VOCAB_FILE, "w", encoding="utf-8") as f:
        f.write("<PAD>:0\n")
        f.write("<UNK>:1\n")
        f.write("<EOS>:2\n")

    print(f"✨ Всё готово! Файлы успешно обновлены под требования вашей модели.")

if __name__ == "__main__":
    main()
