import json

from typing import Dict
from pathlib import Path
from operator import itemgetter
from wordcloud import WordCloud, STOPWORDS


def get_wordcloud_data(transcript_text: str) -> Dict[str, int]:
    with open(Path(__file__).resolve().parent / "hi_stopwords.json", encoding="utf-8") as f:
        hindi_stopwords = json.load(f)

    with open(Path(__file__).resolve().parent / "en_stopwords.json", encoding="utf-8") as f:
        english_stopwords = json.load(f)

    # STOPWORDS is a set
    STOPWORDS.update(hindi_stopwords)
    STOPWORDS.update(english_stopwords)
    wordcloud_data = WordCloud(stopwords=STOPWORDS).process_text(transcript_text)
    data = []
    for word, count in wordcloud_data.items():
        data.append({"text": word, "count": count})

    data.sort(key=itemgetter("count"), reverse=True)
    return data[:20]
