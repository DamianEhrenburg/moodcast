import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import pipeline

from .config import Config
from .models import EmotionalState, EmotionType

logger = logging.getLogger(__name__)

EMOTION_GENRE_MAPPING = {
    EmotionType.EXCITED: {
        "genres": ["боевик", "приключения", "фантастика", "триллер"],
        "excluded": ["драма", "мелодрама"],
        "min_rating": 7.0,
        "year_range": 30,
    },
    EmotionType.TENSE: {
        "genres": ["триллер", "детектив", "ужасы", "криминал"],
        "excluded": ["комедия", "семейный", "мелодрама"],
        "min_rating": 7.0,
        "year_range": 40,
    },
    EmotionType.HAPPY: {
        "genres": ["комедия", "семейный", "приключения", "мелодрама"],
        "excluded": ["ужасы", "триллер", "драма"],
        "min_rating": 7.0,
        "year_range": 30,
    },
    EmotionType.CALM: {
        "genres": ["драма", "мелодрама", "биография", "документальный"],
        "excluded": ["боевик", "ужасы", "триллер"],
        "min_rating": 6.8,
        "year_range": 50,
    },
    EmotionType.SAD: {
        "genres": ["драма", "мелодрама"],
        "excluded": ["комедия", "боевик", "семейный"],
        "min_rating": 7.0,
        "year_range": 50,
    },
    EmotionType.BORED: {
        "genres": ["приключения", "фантастика", "комедия", "боевик", "триллер"],
        "excluded": ["драма", "документальный"],
        "min_rating": 6.8,
        "year_range": 30,
    },
    EmotionType.INSPIRED: {
        "genres": ["биография", "драма", "история", "документальный"],
        "excluded": ["ужасы", "комедия", "боевик"],
        "min_rating": 7.2,
        "year_range": 50,
    },
    EmotionType.ROMANTIC: {
        "genres": ["мелодрама", "комедия", "драма"],
        "excluded": ["ужасы", "боевик", "триллер"],
        "min_rating": 6.8,
        "year_range": 40,
    },
    EmotionType.THOUGHTFUL: {
        "genres": ["драма", "детектив", "фантастика", "биография"],
        "excluded": ["комедия", "боевик"],
        "min_rating": 7.0,
        "year_range": 50,
    },
    EmotionType.NEUTRAL: {
        "genres": [],
        "excluded": [],
        "min_rating": 7.0,
        "year_range": 40,
    },
    EmotionType.HORROR: {
        "genres": ["ужасы", "триллер"],
        "excluded": ["комедия", "мелодрама", "семейный"],
        "min_rating": 6.0,
        "year_range": 40,
    },
    EmotionType.SARCASTIC: {
        "genres": ["комедия", "драма"],
        "excluded": ["семейный", "мелодрама"],
        "min_rating": 7.0,
        "year_range": 30,
    },
    EmotionType.ANGRY: {
        "genres": ["боевик", "триллер", "криминал", "драма"],
        "excluded": ["мелодрама", "комедия", "семейный"],
        "min_rating": 6.5,
        "year_range": 40,
    },
    EmotionType.CONFUSED: {
        "genres": ["детектив", "фантастика", "триллер", "драма"],
        "excluded": ["комедия"],
        "min_rating": 7.0,
        "year_range": 40,
    },
    EmotionType.SURPRISED: {
        "genres": ["триллер", "приключения", "фантастика", "детектив", "комедия"],
        "excluded": ["драма", "мелодрама"],
        "min_rating": 7.0,
        "year_range": 30,
    },
}

EMOTION_LABELS_PHRASES = {
    "excited": "автор хочет посмотреть что-то захватывающее, динамичное, энергичное или бодрящее",
    "tense": "пользователь ищет напряженный, тревожный, остросюжетный фильм или триллер",
    "happy": "сообщение говорит о хорошем настроении, желании посмотреть легкую комедию или что-то веселое",
    "calm": "автор хочет спокойный, умиротворяющий или расслабляющий контент",
    "sad": "человеку грустно, он ищет меланхоличный фильм, драму или что-то созвучное настроению",
    "bored": "сообщение передает скуку, нужно что-то увлекательное, захватывающее или чтобы отвлечься",
    "inspired": "текст говорит о вдохновении, мотивации или интересе к биографиям и историям успеха",
    "romantic": "пользователь ищет романтику, фильм про любовь или для просмотра вдвоем",
    "thoughtful": "автор настроен задумчиво, ищет что-то глубокое, философское или заставляющее подумать",
    "neutral": "сообщение нейтральное, общий запрос без явной эмоции",
    "horror": "пользователь хочет посмотреть ужасы, что-то страшное или пугающее",
    "sarcastic": "сообщение содержит сарказм, иронию или черный юмор",
    "angry": "автор злится, хочет что-то жесткое, возможно, боевик или кровавый фильм",
    "confused": "пользователь в замешательстве, ищет что-то запутанное, детектив или головоломку",
    "surprised": "текст выражает удивление, интерес к неожиданным поворотам сюжета",
}

_classifier_pipeline = None


def get_emotion_pipeline():
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline

    try:
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        logger.info(f"Loading Zero-Shot NLP model: {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        _classifier_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model loaded successfully on {device_name}")
    except Exception as e:
        logger.critical(f"Failed to load ML model: {e}", exc_info=True)
        _classifier_pipeline = None

    return _classifier_pipeline


class EmotionAnalyzer:
    def __init__(self, cache_path: str = "emotion_cache.json"):
        self.cache_path = Path(cache_path)
        self._emotion_cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self._emotion_cache = json.loads(content)
                        logger.info(f"Loaded emotion cache ({len(self._emotion_cache)} entries)")
        except Exception as e:
            logger.warning(f"Could not load emotion cache: {e}")
            self._emotion_cache = {}

    async def _save_cache(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_cache_sync)
        except Exception as e:
            logger.warning(f"Async cache save error: {e}")

    def _save_cache_sync(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._emotion_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Sync cache save error: {e}")

    @staticmethod
    def clean_text(text: str) -> str:
        current = text.strip()
        if not current:
            return ""
        cleaned = re.sub(r"http\S+|www\S+", "", current, flags=re.MULTILINE)
        cleaned = re.sub(r"@\w+", "", cleaned)
        cleaned = cleaned.replace("\n", " ").strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned

    async def analyze(self, text: str) -> Optional[EmotionalState]:
        nlp_pipeline = get_emotion_pipeline()
        if not nlp_pipeline:
            logger.error("Emotion analysis unavailable: pipeline not initialized.")
            return None

        processed = self.clean_text(text)
        if not processed:
            return None

        cache_key = processed
        if cache_key in self._emotion_cache:
            try:
                cached = self._emotion_cache[cache_key]
                key = cached["emotion_key"]
                intensity = cached["intensity"]
                if key.upper() in EmotionType.__members__:
                    emo_type = EmotionType[key.upper()]
                    mapping = EMOTION_GENRE_MAPPING[emo_type]
                    return EmotionalState(
                        primary_emotion=key,
                        intensity=intensity,
                        suggested_genres=mapping.get("genres", []),
                        excluded_genres=mapping.get("excluded", []),
                    )
            except Exception:
                del self._emotion_cache[cache_key]

        try:
            candidate_labels = list(EMOTION_LABELS_PHRASES.keys())
            hypothesis = "Эмоция этого текста - {}."

            result = nlp_pipeline(
                processed,
                candidate_labels,
                hypothesis_template=hypothesis,
                multi_label=False,
            )

            primary_key = result["labels"][0]
            intensity = result["scores"][0]

            final_key = primary_key
            final_intensity = intensity

            if primary_key and intensity < Config.MIN_EMOTION_CONFIDENCE:
                final_key = EmotionType.NEUTRAL.name.lower()
                final_intensity = 0.0

            if not final_key or final_key.upper() not in EmotionType.__members__:
                emo_type = EmotionType.NEUTRAL
                final_key = EmotionType.NEUTRAL.name.lower()
            else:
                emo_type = EmotionType[final_key.upper()]

            mapping = EMOTION_GENRE_MAPPING[emo_type]

            self._emotion_cache[cache_key] = {
                "emotion_key": final_key,
                "intensity": final_intensity,
            }
            asyncio.create_task(self._save_cache())

            return EmotionalState(
                primary_emotion=final_key,
                intensity=final_intensity,
                suggested_genres=mapping.get("genres", []),
                excluded_genres=mapping.get("excluded", []),
            )
        except Exception as e:
            logger.error(f"Error analyzing text: {e}", exc_info=True)
            return None
