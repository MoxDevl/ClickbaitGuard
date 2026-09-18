import csv
import html
import re
import time
import feedparser   # pip installed
import trafilatura  # pip installed
from tqdm import tqdm


RSS_FEEDS = [
    "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "https://lenta.ru/rss/news",
    "https://www.interfax.ru/rss.asp",
    "https://www.kommersant.ru/RSS/news.xml",
]

TARGET_COUNT = 500
OUTPUT_FILE = "news_corpus_500(1).csv"


def remove_title_from_text(text: str, title: str) -> str:
    if not text or not title:
        return text

    clean_title = title.strip(' "«»„“')
    if not clean_title:
        return text

    pattern = rf"^\s*{re.escape(clean_title)}[ .:—\-]*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def collect_news(target_count=500):
    collected_articles = []
    seen_urls = set()

    print("Сбор ссылок из новостных лент...")
    candidate_entries = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            link = getattr(entry, "link", None)
            title = getattr(entry, "title", "").strip()

            if link and title and link not in seen_urls:
                seen_urls.add(link)
                candidate_entries.append(
                    {"title": html.unescape(title), "url": link}
                )

    print(f"Найдено уникальных кандидатов: {len(candidate_entries)}")
    print("Извлечение и очистка полного текста статей...")

    for item in tqdm(candidate_entries):
        if len(collected_articles) >= target_count:
            break

        url = item["url"]
        title = item["title"]

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue

            full_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )

            if not full_text:
                continue

            cleaned_text = re.sub(r"\s+", " ", full_text).strip()

            cleaned_text = remove_title_from_text(cleaned_text, title)

            # 3. Проверяем объем уже очищенного текста (не менее 40 слов)
            if len(cleaned_text.split()) >= 40:
                collected_articles.append(
                    {
                        "id": len(collected_articles) + 1,
                        "title": title,
                        "text": cleaned_text,
                        "url": url,
                    }
                )

            time.sleep(0.1)

        except Exception:
            continue

    return collected_articles


articles = collect_news(TARGET_COUNT)

with open(OUTPUT_FILE, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "title", "text", "url"])
    writer.writeheader()
    writer.writerows(articles)

print(f"\nГотово! Собрано {len(articles)} записей. Файл сохранен: {OUTPUT_FILE}")