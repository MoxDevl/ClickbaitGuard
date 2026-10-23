import pandas as pd
import pymorphy3
import re

morph = pymorphy3.MorphAnalyzer()
SKIP_POS = {'NPRO', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NUMR'}

# СТОП-СЛОВА
NEWS_STOP_WORDS = {
    'год', 'рубль', 'россиянин', 'быть', 'весь', 'который', 'этот', 
    'ждать', 'получить', 'сделать', 'знать', 'стать', 'какой', 'деньга',
    'сразу', 'только', 'уже', 'каждый', 'россия', 'составить', 'дать'
}

def load_lexicon(lexicon_path):
    print(f"Загрузка словаря из {lexicon_path}...")
    df = pd.read_csv(lexicon_path, encoding='utf-8-sig')
    lexicon = {}
    for _, row in df.iterrows():
        word = row['word']
        if word in NEWS_STOP_WORDS:
            continue
        if word not in lexicon or row['importance'] > lexicon[word]['weight']:
            lexicon[word] = {'weight': row['importance'], 'type': row['feature_type']}
    return lexicon

def get_morph_info(text):
    text = str(text)
    raw_tokens = re.findall(r'[А-Яа-яЁё]+', text)
    
    morph_info = []
    for i, token in enumerate(raw_tokens):
        if len(token) <= 2:
            continue
        if i > 0 and token[0].isupper():
            continue
            
        token_lower = token.lower()
        if token_lower in NEWS_STOP_WORDS:
            continue

        p = morph.parse(token_lower)[0]
        
        # Прошедшего времени (нашли, отказалась)
        if token_lower.endswith(('ли', 'ла', 'ло', 'л', 'лась', 'лось', 'лись', 'лся')) and p.tag.POS == 'VERB':
            parses = morph.parse(token_lower)
            for parse in parses:
                if 'past' in parse.tag:
                    p = parse
                    break
        
        if p.tag.POS in SKIP_POS:
            continue
            
        mood = p.tag.mood
        degree = 'Supr' if 'Supr' in p.tag else None
        morph_info.append((p.normal_form, p.tag.POS, mood, degree))
        
    return morph_info

def calculate_clickbait_score(headline, lexicon):
    words_info = get_morph_info(headline)
    score = 0.0
    matched_words = []
    
    for lemma, pos, mood, degree in words_info:
        if lemma in lexicon:
            info = lexicon[lemma]
            base_weight = info['weight']
            f_type = info['type']
            
            if f_type == 'imperative_verb' and mood == 'impr':
                score += base_weight * 3.0
                matched_words.append(f"{lemma} (imperative)")
            elif f_type == 'superlative_adj' and degree == 'Supr':
                score += base_weight * 3.0
                matched_words.append(f"{lemma} (superlative)")
            else:
                score += base_weight
                matched_words.append(f"{lemma} ({f_type})")
                
    return score, matched_words

def classify_news_file(input_csv, lexicon_path, output_csv, title_col='title', threshold=0.02):
    lexicon = load_lexicon(lexicon_path)
    
    print(f"Чтение новостей из {input_csv}...")
    df = pd.read_csv(input_csv, encoding='utf-8-sig', on_bad_lines='skip')
    
    if title_col not in df.columns:
        print(f"Ошибка: Колонки '{title_col}' нет в файле!")
        return
        
    results = []
    
    print(f"Анализ заголовков (порог: {threshold})...")
    for index, row in df.iterrows():
        title = row[title_col]
        score, matched = calculate_clickbait_score(title, lexicon)
        
        is_clickbait = 1 if score >= threshold else 0
        
        results.append({
            'title': title,
            'score': round(score, 4),
            'is_clickbait': is_clickbait,
            'matched_words': ", ".join(matched)
        })
        
    result_df = pd.DataFrame(results).sort_values(by='score', ascending=False)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    clickbaits = result_df[result_df['is_clickbait'] == 1]
    
    print(f"\nВсе {len(result_df)} заголовков сохранены в {output_csv}")
    print(f"\n--- Найденные кликбейтные заголовки (порог {threshold}) ---")
    if clickbaits.empty:
        print("Ни один заголовок не прошел порог. Это нормально, если база состоит из качественных СМИ.")
    else:
        for i, row in clickbaits.iterrows():
            print(f"[{row['score']}] {row['title']}")
            if row['matched_words']:
                print(f"   ↳ Слова: {row['matched_words']}")

if __name__ == "__main__":
    LEXICON_FILE = "public/results/clickbait_lexicon.csv" 
    INPUT_NEWS = "public/news_corpus.csv" 
    OUTPUT_FILE = "public/results/classified_news.csv"
    
    classify_news_file(
        input_csv=INPUT_NEWS, 
        lexicon_path=LEXICON_FILE, 
        output_csv=OUTPUT_FILE, 
        title_col='title', 
        threshold=0.02
    )