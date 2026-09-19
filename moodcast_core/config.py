import os
from typing import Dict, List, Set
from dotenv import load_dotenv

load_dotenv()

raw_keys = os.getenv("KINOPOISK_API_KEY", "")
KINOPOISK_API_KEYS: List[str] = [k.strip() for k in raw_keys.split(",") if k.strip()]
KINOPOISK_API_KEY: str = KINOPOISK_API_KEYS[0] if KINOPOISK_API_KEYS else ""
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

DEFAULT_POSTER_URL = "https://via.placeholder.com/300x450?text=No+Poster+Available"

GENRE_MAP: Dict[str, int] = {
    "триллер": 1,
    "драма": 2,
    "криминал": 3,
    "мелодрама": 4,
    "детектив": 5,
    "фантастика": 6,
    "приключения": 7,
    "биография": 8,
    "фильм-нуар": 9,
    "вестерн": 10,
    "боевик": 11,
    "фэнтези": 12,
    "комедия": 13,
    "военный": 14,
    "история": 15,
    "музыка": 16,
    "ужасы": 17,
    "мультфильм": 18,
    "семейный": 19,
    "мюзикл": 20,
    "спорт": 21,
    "документальный": 22,
    "короткометражка": 23,
    "аниме": 24,
    "для взрослых": 28,
    "детский": 33,
}


class Config:
    API_BASE_URL: str = "https://kinopoiskapiunofficial.tech/api/v2.2"
    FILMS_URL: str = f"{API_BASE_URL}/films"
    COLLECTIONS_URL: str = f"{API_BASE_URL}/films/collections"

    REQUEST_TIMEOUT: int = 6
    MAX_RETRIES: int = 1

    MIN_RATING: float = 6.5
    MAX_RECOMMENDATIONS: int = 200
    MAX_PAGES: int = 2
    N_REC_TO_SHUFFLE: int = 15

    PRIORITY_COUNTRIES: Dict[str, float] = {
        "сша": 2.5, "великобритания": 2.2, "германия": 1.8, "франция": 1.8,
        "канада": 1.8, "италия": 1.6, "испания": 1.6, "австралия": 1.6,
        "ирландия": 1.5, "дания": 1.4, "швеция": 1.4, "норвегия": 1.4,
    }
    ALLOWED_COUNTRIES_FOR_POST_FILTER: Set[str] = set(PRIORITY_COUNTRIES.keys()) | {"россия", "южная корея", "япония"}

    EXCLUDED_GENRES: Set[str] = {
        "документальный", "короткометражка", "для взрослых",
        "аниме", "ток-шоу", "реальное тв", "новости",
        "концерт", "церемония", "игра"
    }

    MIN_EMOTION_CONFIDENCE: float = 0.30
