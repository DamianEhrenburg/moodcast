import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .models import ContentType

logger = logging.getLogger(__name__)

GENRE_PATTERNS: Dict[str, str] = {
    "документальный": r"\b(?:документ\w*|научпоп\w*|научно-популярн\w*)",
    "комедия": r"\b(?:комед\w*|смешн\w*|ржачн\w*|поржать|юмор\w*|весел\w*)",
    "ужасы": r"\b(?:ужас\w*|ужастик\w*|хоррор\w*|страшн\w*|пугающ\w*)",
    "триллер": r"\b(?:триллер\w*|остросюжетн\w*|саспенс\w*)",
    "фантастика": r"\b(?:фантастик\w*|сайфай|сай-фай|sci-fi)",
    "детектив": r"\b(?:детектив\w*|расследован\w*|сыщик\w*)",
    "боевик": r"\b(?:боевик\w*|экшен\w*|экшн\w*|боевичек)",
    "мелодрама": r"\b(?:мелодрам\w*|романти\w*|про\s+любовь|лавстори)",
    "драма": r"\b(?:драм\w*|драматическ\w*)",
    "приключения": r"\b(?:приключен\w*)",
    "криминал": r"\b(?:криминал\w*|про\s+мафию|про\s+бандитов|гангстер\w*)",
    "биография": r"\b(?:биографи\w*|байопик\w*|на\s+реальных\s+событиях)",
    "военный": r"\b(?:военн\w*|про\s+войну|про\s+фронт|великая\s+отечественная|вторая\s+мировая)",
    "история": r"\b(?:историч\w*|эпох\w*|средневеков\w*)",
    "фэнтези": r"\b(?:фэнтези|фэнтази|про\s+магов|волшебств\w*)",
    "вестерн": r"\b(?:вестерн\w*|дикий\s+запад|ковбой\w*)",
    "спорт": r"\b(?:спортивн\w*|про\s+спорт|про\s+футбол|про\s+бокс|про\s+хоккей|про\s+баскетбол)",
    "музыка": r"\b(?:музыкальн\w*|мюзикл\w*|про\s+музыкант\w*)",
    "мультфильм": r"\b(?:мульт\w*|мультик\w*|анимац\w*)",
    "семейный": r"\b(?:семейн\w*|с\s+детьми|для\s+всей\s+семьи|детск\w*)",
    "аниме": r"\b(?:аниме|анимэ)",
    "короткометражка": r"\b(?:короткометр\w*|короткий\s+метр)",
    "для взрослых": r"\b(?:18\+|для\s+взрослых|эротик\w*)",
}

TOPIC_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(?:супергеро\w*|marvel|марвел|dc|диси)\b", "супергерои"),
    (r"\b(?:космос\w*|межзвездн\w*|астронавт\w*)\b", "космос"),
    (r"\b(?:зомби|мертвец\w*|апокалипсис\s+зомби)\b", "зомби"),
    (r"\b(?:вампир\w*|дракул\w*)\b", "вампиры"),
    (r"\b(?:выживан\w*|в\s+дикой\s+природе|на\s+необитаемом)\b", "выживание"),
    (r"\b(?:ограблен\w*|грабител\w*|куш)\b", "ограбление"),
    (r"\b(?:маньяк\w*|серийн\w*\s+убийц\w*)\b", "маньяки"),
    (r"\b(?:путешестви\w*\s+во\s+времени|петл\w*\s+времени)\b", "путешествия во времени"),
    (r"\b(?:искусственн\w*\s+интеллект\w*|\bии\b|нейросет\w*|робот\w*|киборг\w*)\b", "роботы"),
    (r"\b(?:постапокалипс\w*|после\s+конца\s+света|ядерн\w*\s+войн\w*)\b", "постапокалипсис"),
    (r"\b(?:викинг\w*|вальгалл\w*)\b", "викинги"),
    (r"\b(?:шпион\w*|спецагент\w*|разведк\w*|цру|кгб|ми-6)\b", "шпионы"),
    (r"\b(?:мафи\w*|гангстер\w*|якудза|картел\w*)\b", "гангстеры"),
    (r"\b(?:хакер\w*|киберпреступ\w*)\b", "хакеры"),
    (r"\b(?:шахмат\w*)\b", "шахматы"),
    (r"\b(?:врач\w*|медицин\w*|доктор\w*|больниц\w*)\b", "медицина"),
    (r"\b(?:тюрьм\w*|побег\w*|заключенн\w*|колони\w*)\b", "побег из тюрьмы"),
    (r"\b(?:самура\w*)\b", "самураи"),
    (r"\b(?:собак\w*|животн\w*|кошк\w*)\b", "животные"),
    (r"\b(?:полици\w*|коп\w*|шериф\w*)\b", "полиция"),
]


@dataclass
class ParsedUserIntent:
    explicit_genres: List[str] = field(default_factory=list)
    excluded_genres: Set[str] = field(default_factory=set)
    content_type_override: Optional[str] = None
    topic_keyword: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    allow_animation: bool = False
    allow_adult: bool = False
    allow_documentary: bool = False
    allow_short: bool = False


def parse_user_intent(text: str) -> ParsedUserIntent:
    intent = ParsedUserIntent()
    text_clean = text.lower().strip()

    # 1. Negative filters: "не ужасы", "без комедий", "только не драму"
    neg_matches = re.findall(r"\b(?:не|без|кроме|исключ(?:и|ить|ая))\s+([а-яёa-z0-9\-]+)", text_clean)
    for word in neg_matches:
        for genre, pattern in GENRE_PATTERNS.items():
            if re.search(pattern, word):
                intent.excluded_genres.add(genre)

    # 2. Explicit genres (only if not negated)
    for genre, pattern in GENRE_PATTERNS.items():
        if genre in intent.excluded_genres:
            continue
        # Search for genre pattern outside negation
        if re.search(pattern, text_clean):
            # Verify it wasn't part of a negation
            neg_context = re.search(rf"\b(?:не|без|кроме|исключ\w*)\s+{pattern}", text_clean)
            if not neg_context:
                intent.explicit_genres.append(genre)

    # 3. Category allowances based on explicit intent
    if "документальный" in intent.explicit_genres or re.search(r"\b(?:документ|научпоп)", text_clean):
        intent.allow_documentary = True
        if "документальный" not in intent.explicit_genres:
            intent.explicit_genres.append("документальный")

    if "мультфильм" in intent.explicit_genres or "семейный" in intent.explicit_genres:
        intent.allow_animation = True

    if "аниме" in intent.explicit_genres or re.search(r"\b(?:аниме|анимэ)", text_clean):
        intent.allow_animation = True
        intent.explicit_genres.append("аниме")

    if "для взрослых" in intent.explicit_genres or re.search(r"\b(?:18\+|для\s+взрослых|эротик)", text_clean):
        intent.allow_adult = True
        if "для взрослых" not in intent.explicit_genres:
            intent.explicit_genres.append("для взрослых")

    if "короткометражка" in intent.explicit_genres or re.search(r"\b(?:короткометр|короткий\s+метр)", text_clean):
        intent.allow_short = True
        if "короткометражка" not in intent.explicit_genres:
            intent.explicit_genres.append("короткометражка")

    # 4. Content type override
    if re.search(r"\b(?:сериал\w*|сериальчик\w*|мини-сериал\w*|сезоны|многосерийный)\b", text_clean):
        intent.content_type_override = ContentType.TV.value
    elif re.search(r"\b(?:фильм\w*|фильмец\w*|кино\w*|кинематограф|полный метр)\b", text_clean):
        intent.content_type_override = ContentType.MOVIE.value

    # 5. Era / Decades
    current_year = datetime.now().year
    if re.search(r"\b(?:90-?[ех]|девяност\w*)\b", text_clean):
        intent.year_from, intent.year_to = 1990, 1999
    elif re.search(r"\b(?:80-?[ех]|восьмидесят\w*)\b", text_clean):
        intent.year_from, intent.year_to = 1980, 1989
    elif re.search(r"\b(?:70-?[ех]|семидесят\w*)\b", text_clean):
        intent.year_from, intent.year_to = 1970, 1979
    elif re.search(r"\b(?:2000-?[ех]|нулев\w*)\b", text_clean):
        intent.year_from, intent.year_to = 2000, 2009
    elif re.search(r"\b(?:2010-?[ех]|десят\w*)\b", text_clean):
        intent.year_from, intent.year_to = 2010, 2019
    elif re.search(r"\b(?:новинк\w*|свеж\w*|2023|2024|2025|2026)\b", text_clean):
        intent.year_from, intent.year_to = 2022, current_year
    elif re.search(r"\b(?:классик\w*|старое кино|советск\w*)\b", text_clean):
        intent.year_from, intent.year_to = 1950, 1990

    # 6. Topics and Keywords
    # Check predefined high-confidence topic patterns
    for pattern, normalized_topic in TOPIC_PATTERNS:
        if re.search(pattern, text_clean):
            intent.topic_keyword = normalized_topic
            break

    # If no predefined topic, look for "про / о / об <что-то>"
    if not intent.topic_keyword:
        match = re.search(r"\b(?:про|о|об)\s+([а-яёa-z0-9\-]+(?:\s+[а-яёa-z0-9\-]+)?)", text_clean)
        if match:
            raw_keyword = match.group(1).strip()
            stop_words = {
                "что", "то", "фильм", "сериал", "кино", "жизнь", "любовь",
                "настроение", "вечер", "хороший", "отличный", "какой", "нибудь"
            }
            tokens = [w for w in raw_keyword.split() if w not in stop_words and len(w) > 2]
            if tokens:
                kw = tokens[0]
                if kw.endswith("оев"):
                    kw = kw[:-3] + "ои"
                elif kw.endswith("ов"):
                    kw = kw[:-2] + "ы"
                elif kw.endswith("ев"):
                    kw = kw[:-2] + "и"
                elif kw.endswith("ах"):
                    kw = kw[:-2] + "ы"
                elif kw.endswith("ях"):
                    kw = kw[:-2] + "и"
                elif kw.endswith("ом") and len(kw) > 4:
                    kw = kw[:-2]
                intent.topic_keyword = kw

    logger.info(
        f"Parsed Intent: genres={intent.explicit_genres}, excluded={intent.excluded_genres}, "
        f"type={intent.content_type_override}, topic='{intent.topic_keyword}', "
        f"allow_anim={intent.allow_animation}, allow_doc={intent.allow_documentary}"
    )
    return intent
