import argparse
import os
import re
import pandas as pd
import numpy as np
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

morph = pymorphy3.MorphAnalyzer()
SKIP_POS = {'NPRO', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NUMR'}

def preprocess_and_analyze(text):
    raw_tokens = re.findall(r'[А-Яа-яЁё]+', str(text))
    lemmas = []
    features = {'imperative': [], 'infinitive': [], 'superlative': []}
    
    for i, token in enumerate(raw_tokens):
        if len(token) <= 2: continue
        if i > 0 and token[0].isupper(): continue # Отсекаем имена
            
        token_lower = token.lower()
        p = morph.parse(token_lower)[0]
        
        # Фикс прошедшего времени (нашли -> наслать)
        if token_lower.endswith(('ли', 'ла', 'ло', 'л', 'лась', 'лось', 'лись', 'лся')) and p.tag.POS == 'VERB':
            for parse in morph.parse(token_lower):
                if 'past' in parse.tag:
                    p = parse
                    break
                    
        if p.tag.POS in SKIP_POS: continue
        
        lemma = p.normal_form
        lemmas.append(lemma)
        
        if p.tag.POS == 'VERB' and p.tag.mood == 'impr':
            features['imperative'].append(lemma)
        elif p.tag.POS == 'INFN':
            features['infinitive'].append(lemma)
        elif p.tag.POS in ('ADJF', 'ADJS') and 'Supr' in p.tag:
            features['superlative'].append(lemma)
            
    return " ".join(lemmas), features

def extract_clickbait_features(input_csv, output_csv, title_col, label_col, tfidf_threshold=0.005, morph_threshold=0.02):
    print(f"Загрузка данных из {input_csv}...")
    df = pd.read_csv(input_csv, sep=';', encoding='utf-8-sig')
    
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce').dropna().astype(int)
    cb_df = df[df[label_col] == 1].copy()
    norm_df = df[df[label_col] == 0].copy()
    
    total_cb = len(cb_df)
    total_norm = len(norm_df)
    print(f"Кликбейтов: {total_cb}, Обычных новостей: {total_norm}")
    
    # Предобработка
    cb_df['lemmas'], cb_features = zip(*cb_df[title_col].apply(preprocess_and_analyze))
    norm_df['lemmas'], norm_features = zip(*norm_df[title_col].apply(preprocess_and_analyze))

    output_data = []

    # 1. TF-IDF (разница средних весов)
    all_lemmas = list(cb_df['lemmas']) + list(norm_df['lemmas'])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_lemmas)
    feature_names = vectorizer.get_feature_names_out()
    
    cb_tfidf = tfidf_matrix[:total_cb].mean(axis=0).A1
    norm_tfidf = tfidf_matrix[total_cb:].mean(axis=0).A1
    tfidf_diff = cb_tfidf - norm_tfidf
    
    for word, score in zip(feature_names, tfidf_diff):
        if score > tfidf_threshold:
            output_data.append({'word': word, 'feature_type': 'tfidf_word', 'importance': round(float(score), 4)})
            
    # 2. Морфологические признаки (ВЫЧИТАЕМ ФОН)
    # Собираем все слова из кликбейтов
    cb_imp = [w for d in cb_features for w in d['imperative']]
    cb_inf = [w for d in cb_features for w in d['infinitive']]
    cb_sup = [w for d in cb_features for w in d['superlative']]
    
    # Собираем все слова из обычных новостей
    norm_imp = [w for d in norm_features for w in d['imperative']]
    norm_inf = [w for d in norm_features for w in d['infinitive']]
    norm_sup = [w for d in norm_features for w in d['superlative']]


    def process_morph(cb_list, norm_list, f_type):
        cb_counts = Counter(cb_list)
        norm_counts = Counter(norm_list)
        all_words = set(cb_counts.keys()).union(set(norm_counts.keys()))
        
        for word in all_words:
            # Считаем долю (частоту) в каждом классе
            freq_cb = cb_counts[word] / total_cb
            freq_norm = norm_counts[word] / total_norm if total_norm > 0 else 0
            
            # Разница: насколько чаще слово встречается в кликбейте
            diff = freq_cb - freq_norm
            
            # ИЗМЕНЕНИЕ: Снижаем порог с 0.02 до 0.001 (0.1%), 
            # чтобы ловить редкие, но важные слова (проверьте, узнайте)
            if diff > 0.001: 
                output_data.append({'word': word, 'feature_type': f_type, 'importance': round(diff, 4)})


    process_morph(cb_imp, norm_imp, 'imperative_verb')
    process_morph(cb_inf, norm_inf, 'infinitive_verb')
    process_morph(cb_sup, norm_sup, 'superlative_adj')

    result_df = pd.DataFrame(output_data).sort_values(by=['feature_type', 'importance'], ascending=[True, False])
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Готово! Словарь сохранен в {output_csv} ({len(result_df)} записей).")

if __name__ == "__main__":
    # 1. Сначала создаем парсер и добавляем аргументы
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="public/titles_data.csv")
    parser.add_argument("--output", "-o", default="public/results/clickbait_lexicon.csv")
    parser.add_argument("--title_col", "-t", default="titles")
    parser.add_argument("--label_col", "-l", default="target")
    
    # 2. Только теперь считываем аргументы из командной строки
    args = parser.parse_args()
    
    # 3. И только здесь вызываем функцию, передавая сниженный порог
    extract_clickbait_features(
        input_csv=args.input, 
        output_csv=args.output, 
        title_col=args.title_col, 
        label_col=args.label_col,
        morph_threshold=0.001
    )