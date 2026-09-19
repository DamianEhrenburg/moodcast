import asyncio
import logging
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

from .config import Config, GENRE_MAP, KINOPOISK_API_KEY, KINOPOISK_API_KEYS
from .intent import ParsedUserIntent, parse_user_intent
from .models import ContentType, EmotionalState, EmotionType, UserState
from .nlp import EMOTION_GENRE_MAPPING

logger = logging.getLogger(__name__)


class KinopoiskQuotaExceededError(Exception):
    """Raised when all configured Kinopoisk API keys have hit their daily limit (HTTP 402)."""
    pass


# Themed collections supported directly by Kinopoisk unofficial API
TOPIC_COLLECTIONS: Dict[str, str] = {
    "супергерои": "COMICS_THEME",
    "зомби": "ZOMBIE_THEME",
    "вампиры": "VAMPIRE_THEME",
    "выживание": "CATASTROPHE_THEME",
    "катастрофы": "CATASTROPHE_THEME",
    "роботы": "PROGRAMMERS_THEME",
    "хакеры": "PROGRAMMERS_THEME",
    "программисты": "PROGRAMMERS_THEME",
}

# Verified Kinopoisk IDs of iconic series for topics without native TV collections
TOPIC_SERIES: Dict[str, List[int]] = {
    "супергерои": [
        460586,   # Пацаны (The Boys)
        817509,   # Сорвиголова (Daredevil)
        1421587,  # Миротворец (Peacemaker)
        1203039,  # Локи (Loki)
        978853,   # Каратель (The Punisher)
        804748,   # Готэм (Gotham)
        485542,   # Отбросы (Misfits)
        463401,   # Академия «Амбрелла» (The Umbrella Academy)
        1203040,  # Ванда/Вижн (WandaVision)
        661938,   # Стрела (Arrow)
        1144179,  # Роковой патруль (Doom Patrol)
        1431133,  # Поколение «Ви» (Gen V)
        817506,   # Флэш (The Flash)
        817508,   # Джессика Джонс (Jessica Jones)
        1320559,  # Лунный рыцарь (Moon Knight)
        1254069,  # Соколиный глаз (Hawkeye)
        1046272,  # Хранители (Watchmen)
        1316617,  # Супермен и Лоис (Superman and Lois)
        461353,   # Проповедник (Preacher)
        701756,   # Агенты «Щ.И.Т.» (Agents of S.H.I.E.L.D.)
        939002,   # Легион (Legion)
    ],
}


class RecommendationEngine:
    """Handles fetching, multi-factor scoring, and filtering of Kinopoisk titles."""

    def __init__(self):
        self.api_cache: Dict[str, Optional[List[Dict]]] = {}
        self.api_keys: List[str] = list(KINOPOISK_API_KEYS)
        self.current_key_idx: int = 0
        self.exhausted_keys: Set[str] = set()
        self.emotion_profiles: Dict[EmotionType, Dict] = {
            EmotionType.EXCITED: {"rating_boost_factor": 0.3},
            EmotionType.TENSE: {"rating_boost_factor": 0.25},
            EmotionType.BORED: {"rating_boost_factor": 0.3},
            EmotionType.ANGRY: {"rating_boost_factor": 0.2},
            EmotionType.HORROR: {"rating_boost_factor": 0.2},
            EmotionType.SURPRISED: {"rating_boost_factor": 0.15},
            EmotionType.HAPPY: {"rating_boost_factor": 0.1},
            EmotionType.INSPIRED: {"rating_boost_factor": 0.15},
            EmotionType.ROMANTIC: {"rating_boost_factor": 0.05},
            EmotionType.THOUGHTFUL: {"rating_boost_factor": 0.05},
            EmotionType.SAD: {"rating_boost_factor": 0.0},
            EmotionType.CALM: {"rating_boost_factor": 0.0},
            EmotionType.NEUTRAL: {"rating_boost_factor": 0.0},
            EmotionType.CONFUSED: {"rating_boost_factor": 0.05},
            EmotionType.SARCASTIC: {"rating_boost_factor": 0.0},
        }

    @property
    def current_api_key(self) -> str:
        if not self.api_keys:
            return KINOPOISK_API_KEY
        return self.api_keys[self.current_key_idx]

    def _get_headers(self) -> Dict[str, str]:
        return {
            "X-API-KEY": self.current_api_key,
            "accept": "application/json",
        }

    def _mark_key_exhausted(self) -> None:
        if not self.api_keys:
            raise KinopoiskQuotaExceededError("Daily Kinopoisk API quota exhausted.")
        exhausted_key = self.current_api_key
        self.exhausted_keys.add(exhausted_key)
        logger.warning(
            f"Kinopoisk key ...{exhausted_key[-6:]} hit daily limit (402). "
            f"Exhausted: {len(self.exhausted_keys)}/{len(self.api_keys)}"
        )
        for idx, key in enumerate(self.api_keys):
            if key not in self.exhausted_keys:
                self.current_key_idx = idx
                logger.info(f"Switched to next Kinopoisk API key: ...{key[-6:]}")
                return
        raise KinopoiskQuotaExceededError("All Kinopoisk API keys have exhausted their daily limit.")

    async def get_recommendations(
        self,
        emotional_state: EmotionalState,
        content_type: str,
        user_state: UserState,
        topic_keyword: Optional[str] = None,
        user_prompt: str = "",
        intent: Optional[ParsedUserIntent] = None,
    ) -> List[Dict]:
        try:
            if intent is None:
                intent = parse_user_intent(user_prompt)

            emotion_key = emotional_state.primary_emotion
            if emotion_key.upper() not in EmotionType.__members__:
                logger.error(f"Invalid emotion key: {emotion_key}")
                return []
            emotion_type = EmotionType[emotion_key.upper()]

            active_excluded = set(Config.EXCLUDED_GENRES)

            # Animation guard: suppress cartoons unless explicitly requested
            if not intent.allow_animation:
                active_excluded.add("мультфильм")
                active_excluded.add("детский")
            else:
                active_excluded.discard("мультфильм")
                active_excluded.discard("детский")
                active_excluded.discard("аниме")

            if intent.allow_documentary:
                active_excluded.discard("документальный")

            if intent.allow_adult:
                active_excluded.discard("для взрослых")

            if intent.allow_short:
                active_excluded.discard("короткометражка")

            # Dynamic negative user exclusions
            active_excluded.update(intent.excluded_genres)

            effective_type = intent.content_type_override or content_type
            effective_keyword = intent.topic_keyword or topic_keyword

            api_params = self._build_api_params(
                emotion_type, emotional_state, effective_type, effective_keyword, intent
            )

            async with aiohttp.ClientSession() as session:
                all_candidates = []
                candidate_ids = set()

                # 1. Dedicated curated TV series for topics like "супергерои"
                if effective_keyword and effective_type == ContentType.TV.value and effective_keyword in TOPIC_SERIES:
                    series_ids = TOPIC_SERIES[effective_keyword]
                    unseen_ids = [sid for sid in series_ids if not user_state.is_recommendation_shown(sid)]
                    ids_to_use = unseen_ids if unseen_ids else series_ids
                    shuffled_ids = list(ids_to_use)
                    random.shuffle(shuffled_ids)
                    for sid in shuffled_ids:
                        candidate_ids.add(sid)
                        all_candidates.append({
                            "kinopoiskId": sid,
                            "type": "TV_SERIES",
                            "ratingKinopoisk": 8.2,
                            "ratingImdb": 8.4,
                            "genres": [{"genre": "фантастика"}, {"genre": "боевик"}],
                            "countries": [{"country": "сша"}],
                        })

                # 2. Official Kinopoisk themed collections for movies
                elif effective_keyword and effective_type != ContentType.TV.value and effective_keyword in TOPIC_COLLECTIONS:
                    collection_name = TOPIC_COLLECTIONS[effective_keyword]
                    collection_items = await self._fetch_collection(session, collection_name)
                    for item in collection_items:
                        fid = item.get("kinopoiskId")
                        if fid and fid not in candidate_ids and not user_state.is_recommendation_shown(fid):
                            candidate_ids.add(fid)
                            all_candidates.append(item)

                # 3. Standard search across Kinopoisk catalog if no special collection matched
                if not all_candidates:
                    fetched_pages = 0
                    max_pages_to_fetch = Config.MAX_PAGES

                    for page_num in range(1, max_pages_to_fetch + 1):
                        api_params["page"] = page_num
                        page_items = await self._fetch_recommendations(session, api_params)

                        if page_items is None:
                            continue
                        if not page_items:
                            break

                        for item in page_items:
                            film_id = item.get("kinopoiskId")
                            if not film_id or film_id in candidate_ids or user_state.is_recommendation_shown(film_id):
                                continue

                            rating_val = item.get("ratingKinopoisk") or item.get("rating")
                            try:
                                rating_float = float(rating_val) if rating_val is not None else 0.0
                            except (ValueError, TypeError):
                                rating_float = 0.0

                            min_allowed = float(api_params.get("ratingFrom", Config.MIN_RATING))
                            if rating_float < min_allowed:
                                continue

                            candidate_ids.add(film_id)
                            all_candidates.append(item)

                        fetched_pages += 1
                        if len(all_candidates) >= 15:
                            break

                    # If user searched by topic keyword and nothing was found, do NOT silently fall back
                    if not all_candidates and effective_keyword:
                        logger.info(f"No results found for topic keyword: '{effective_keyword}'")
                        return []

                    # Fallback to broader search if emotion search had few results
                    if not all_candidates and not effective_keyword and fetched_pages < 3:
                        fallback_params = api_params.copy()
                        fallback_params.pop("genres", None)
                        fallback_params["page"] = 1
                        fallback_min = max(5.0, Config.MIN_RATING - 1.0)
                        fallback_params["ratingFrom"] = str(fallback_min)

                        page_items = await self._fetch_recommendations(session, fallback_params)
                        if page_items:
                            for item in page_items:
                                film_id = item.get("kinopoiskId")
                                if not film_id or film_id in candidate_ids or user_state.is_recommendation_shown(film_id):
                                    continue
                                candidate_ids.add(film_id)
                                all_candidates.append(item)

                if not all_candidates:
                    return []

                scored = self._score_and_filter(
                    all_candidates,
                    emotion_type,
                    emotional_state,
                    active_excluded,
                    explicit_genres=intent.explicit_genres,
                )
                if not scored:
                    return []

                sorted_scored = sorted(scored, key=lambda x: x[1], reverse=True)
                if len(sorted_scored) > 1:
                    top_to_shuffle = sorted_scored[:Config.N_REC_TO_SHUFFLE]
                    random.shuffle(top_to_shuffle)
                    final_tuples = top_to_shuffle + sorted_scored[Config.N_REC_TO_SHUFFLE:]
                else:
                    final_tuples = sorted_scored

                filtered_by_country = []
                allowed = Config.ALLOWED_COUNTRIES_FOR_POST_FILTER
                for movie, _ in final_tuples:
                    countries = [c.get("country", "").lower() for c in movie.get("countries", [])]
                    if any(c in allowed for c in countries):
                        filtered_by_country.append(movie)

                return (filtered_by_country or [m for m, _ in final_tuples])[:Config.MAX_RECOMMENDATIONS]

        except KinopoiskQuotaExceededError:
            raise
        except Exception as e:
            logger.error(f"Error in get_recommendations: {e}", exc_info=True)
            return []

    def _build_api_params(
        self,
        emotion_type: EmotionType,
        emotional_state: EmotionalState,
        content_type: str,
        topic_keyword: Optional[str] = None,
        intent: Optional[ParsedUserIntent] = None,
    ) -> Dict:
        current_year = datetime.now().year
        emotion_params = EMOTION_GENRE_MAPPING.get(
            emotion_type, EMOTION_GENRE_MAPPING[EmotionType.NEUTRAL]
        )

        effective_keyword = (intent.topic_keyword if intent else None) or topic_keyword
        effective_type = (intent.content_type_override if intent else None) or content_type

        if intent and intent.allow_adult:
            rating_from = "4.5"
        elif intent and intent.allow_short:
            rating_from = "5.0"
        elif effective_keyword:
            rating_from = "6.0"
        else:
            rating_from = str(Config.MIN_RATING)

        params = {
            "order": "RATING",
            "type": "TV_SERIES" if effective_type == ContentType.TV.value else "FILM",
            "ratingFrom": rating_from,
            "ratingTo": "10",
        }

        if intent and intent.year_from:
            params["yearFrom"] = str(intent.year_from)
            params["yearTo"] = str(intent.year_to or current_year)
        elif effective_keyword:
            params["yearFrom"] = "1980"
            params["yearTo"] = str(current_year)
        else:
            params["yearFrom"] = str(current_year - emotion_params["year_range"])
            params["yearTo"] = str(current_year)

        if effective_keyword:
            params["keyword"] = effective_keyword
            logger.info(f"Topic keyword search: '{effective_keyword}' without genre restriction in API")
        elif intent and intent.explicit_genres:
            primary_genre = intent.explicit_genres[0].lower()
            genre_id = GENRE_MAP.get(primary_genre)
            if genre_id:
                params["genres"] = str(genre_id)
                logger.info(f"Explicit genre search: '{primary_genre}' (id: {genre_id})")
        elif emotional_state.suggested_genres:
            primary_genre = emotional_state.suggested_genres[0].lower()
            genre_id = GENRE_MAP.get(primary_genre)
            if genre_id:
                params["genres"] = str(genre_id)

        return params

    async def _fetch_recommendations(
        self, session: aiohttp.ClientSession, params: Dict
    ) -> Optional[List[Dict]]:
        cache_key_params = tuple(sorted((k, v) for k, v in params.items() if k != "page"))
        page_num = params.get("page", 0)
        cache_key = f"{params.get('type', 'any')}_{hash(cache_key_params)}_page{page_num}"

        if cache_key in self.api_cache:
            return self.api_cache[cache_key]

        retries = 0
        while retries <= Config.MAX_RETRIES:
            try:
                headers = self._get_headers()
                async with session.get(
                    Config.FILMS_URL,
                    headers=headers,
                    params=params,
                    timeout=Config.REQUEST_TIMEOUT,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        valid = []
                        for item in items:
                            rating = item.get("ratingKinopoisk")
                            try:
                                float(rating)
                                valid.append(item)
                            except (ValueError, TypeError):
                                pass
                        self.api_cache[cache_key] = valid
                        if len(self.api_cache) > 200:
                            for k in random.sample(list(self.api_cache.keys()), 50):
                                self.api_cache.pop(k, None)
                        return valid

                    elif resp.status == 404:
                        self.api_cache[cache_key] = []
                        return []
                    elif resp.status == 401:
                        logger.error(f"Kinopoisk API 401 Unauthorized for key ...{self.current_api_key[-6:]}")
                        self._mark_key_exhausted()
                        retries += 1
                        continue
                    elif resp.status == 402:
                        self._mark_key_exhausted()
                        retries += 1
                        continue
                    elif resp.status == 429:
                        logger.warning("Kinopoisk rate limit (429). Waiting 2s...")
                        await asyncio.sleep(2)
                        retries += 1
                    else:
                        retries += 1
                        await asyncio.sleep(1)

            except KinopoiskQuotaExceededError:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.warning(f"Network error on Kinopoisk request: {e}")
                retries += 1
                if retries <= Config.MAX_RETRIES:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected Kinopoisk API error: {e}", exc_info=True)
                return None

        return None

    async def _fetch_collection(
        self, session: aiohttp.ClientSession, collection_type: str, max_pages: int = 2
    ) -> List[Dict]:
        items = []
        for page in range(1, max_pages + 1):
            cache_key = f"coll_{collection_type}_page{page}"
            if cache_key in self.api_cache:
                cached = self.api_cache[cache_key]
                if cached:
                    items.extend(cached)
                continue

            try:
                headers = self._get_headers()
                params = {"type": collection_type, "page": page}
                async with session.get(
                    Config.COLLECTIONS_URL,
                    headers=headers,
                    params=params,
                    timeout=Config.REQUEST_TIMEOUT,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        page_items = data.get("items", [])
                        valid = [item for item in page_items if item.get("kinopoiskId")]
                        self.api_cache[cache_key] = valid
                        items.extend(valid)
                        if not page_items or page >= data.get("totalPages", 1):
                            break
                    elif resp.status == 402:
                        self._mark_key_exhausted()
                        break
                    else:
                        break
            except KinopoiskQuotaExceededError:
                raise
            except Exception as e:
                logger.warning(f"Error fetching collection {collection_type} page {page}: {e}")
                break
        return items

    def _score_and_filter(
        self,
        recommendations: List[Dict],
        emotion_type: EmotionType,
        emotional_state: EmotionalState,
        active_excluded: Optional[Set[str]] = None,
        explicit_genres: Optional[List[str]] = None,
    ) -> List[Tuple[Dict, float]]:
        scored = []
        seen = set()
        excluded = active_excluded if active_excluded is not None else Config.EXCLUDED_GENRES
        explicit_set = {g.lower() for g in explicit_genres} if explicit_genres else set()

        for movie in recommendations:
            film_id = movie.get("kinopoiskId")
            if not film_id or film_id in seen:
                continue
            seen.add(film_id)

            genres = {g["genre"].lower() for g in movie.get("genres", []) if "genre" in g}

            # 1. Strict explicit genre requirement (Hard Filter)
            if explicit_set and not genres.intersection(explicit_set):
                continue

            # 2. Excluded genres filter
            if any(ex in genres for ex in excluded):
                continue
            if any(ex.lower() in genres for ex in emotional_state.excluded_genres):
                continue

            # 3. IMDb Discrepancy Filter (Mega Man Guard: rejects toxic / low-voter scores, except for 18+)
            if "для взрослых" not in explicit_set:
                rating_imdb_raw = movie.get("ratingImdb")
                if rating_imdb_raw is not None:
                    try:
                        imdb_val = float(rating_imdb_raw)
                        if imdb_val < 5.5:
                            logger.info(
                                f"Skipping '{movie.get('nameRu')}': IMDb {imdb_val} < 5.5 despite KP {movie.get('ratingKinopoisk')}"
                            )
                            continue
                    except (ValueError, TypeError):
                        pass

            score = self._calculate_relevance_score(
                movie, emotion_type, emotional_state, genres, explicit_genres=explicit_genres
            )
            scored.append((movie, score))

        if not scored and recommendations:
            fallback = []
            for movie in recommendations:
                genres = {g["genre"].lower() for g in movie.get("genres", []) if "genre" in g}
                if explicit_set and not genres.intersection(explicit_set):
                    continue
                if any(ex in genres for ex in excluded):
                    continue
                try:
                    r = float(movie.get("ratingKinopoisk", 0.0))
                except (ValueError, TypeError):
                    r = 0.0
                fallback.append((movie, r))
                if len(fallback) >= 10:
                    break
            return fallback

        return scored

    def _calculate_relevance_score(
        self,
        movie: Dict,
        emotion_type: EmotionType,
        emotional_state: EmotionalState,
        genres: Set[str],
        explicit_genres: Optional[List[str]] = None,
    ) -> float:
        try:
            kp_score = float(movie.get("ratingKinopoisk", 0.0))
        except (ValueError, TypeError):
            kp_score = 0.0

        try:
            imdb_score = float(movie.get("ratingImdb", 0.0)) if movie.get("ratingImdb") else None
        except (ValueError, TypeError):
            imdb_score = None

        if imdb_score is not None and imdb_score > 0:
            base_score = kp_score * 0.6 + imdb_score * 0.4
        else:
            base_score = kp_score

        genre_bonus = 0.0
        if explicit_genres:
            if any(eg.lower() in genres for eg in explicit_genres):
                genre_bonus += 3.5
        elif emotional_state.suggested_genres:
            matches = sum(1 for g in emotional_state.suggested_genres if g.lower() in genres)
            genre_bonus = matches * 1.5

        country_bonus = 0.0
        countries = [c.get("country", "").lower() for c in movie.get("countries", []) if "country" in c]
        best_priority = 0.0
        has_priority = False
        for c in countries:
            if c in Config.PRIORITY_COUNTRIES:
                best_priority = max(best_priority, Config.PRIORITY_COUNTRIES[c])
                has_priority = True

        country_bonus = best_priority if has_priority else -4.0

        if not movie.get("nameOriginal") and not movie.get("nameEn"):
            country_bonus -= 0.5

        profile = self.emotion_profiles.get(emotion_type, {})
        rating_boost = profile.get("rating_boost_factor", 0.0)
        intensity_factor = max(0.0, emotional_state.intensity) ** 0.5
        psychological_bonus = (base_score * rating_boost) * intensity_factor

        year_bonus = 0.0
        try:
            year = int(movie.get("year", 0))
            curr_year = datetime.now().year
            if year >= curr_year - 10:
                year_bonus = 0.4
            elif year >= curr_year - 20:
                year_bonus = 0.1
            elif year < curr_year - 35:
                year_bonus = -0.3
        except (ValueError, TypeError):
            pass

        final_score = (
            base_score * 1.0
            + genre_bonus * 1.2
            + country_bonus * 1.5
            + psychological_bonus * 1.0
            + year_bonus * 0.5
        )
        return max(0.1, final_score)

    async def get_film_details(
        self, film_id: int, session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        cache_key = f"film_details_{film_id}"
        if cache_key in self.api_cache:
            return self.api_cache[cache_key]

        details_url = f"{Config.FILMS_URL}/{film_id}"
        seasons_url = f"{Config.FILMS_URL}/{film_id}/seasons"

        retries = 0
        data = None
        while retries <= Config.MAX_RETRIES:
            try:
                headers = self._get_headers()
                async with session.get(
                    details_url, headers=headers, timeout=Config.REQUEST_TIMEOUT
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        break
                    elif resp.status == 401 or resp.status == 402:
                        self._mark_key_exhausted()
                        retries += 1
                        continue
                    elif resp.status == 429:
                        await asyncio.sleep(1.5)
                        retries += 1
                    else:
                        retries += 1
                        await asyncio.sleep(1.0)
            except KinopoiskQuotaExceededError:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError):
                retries += 1
                if retries <= Config.MAX_RETRIES:
                    await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Error fetching film details for {film_id}: {e}")
                return None

        if not data:
            return None

        # Prefer description, fallback to shortDescription
        if not data.get("description") and data.get("shortDescription"):
            data["description"] = data["shortDescription"]

        length = data.get("filmLength")
        if isinstance(length, int) and length > 0:
            h, m = divmod(length, 60)
            data["filmLengthFormatted"] = f"{h} ч. {m} мин." if h else f"{m} мин."
        else:
            data["filmLengthFormatted"] = None

        if data.get("type") == "TV_SERIES":
            data["seasonsCount"] = "N/A"
            try:
                headers = self._get_headers()
                async with session.get(
                    seasons_url, headers=headers, timeout=Config.REQUEST_TIMEOUT
                ) as s_resp:
                    if s_resp.status == 200:
                        s_json = await s_resp.json()
                        tot = s_json.get("total", 0)
                        if tot == 0 and s_json.get("items"):
                            tot = len(s_json.get("items", []))
                        if isinstance(tot, int) and tot > 0:
                            data["seasonsCount"] = str(tot)
            except Exception:
                pass
        else:
            data["seasonsCount"] = None

        if not data.get("nameOriginal") and data.get("nameEn"):
            data["nameOriginal"] = data["nameEn"]

        self.api_cache[cache_key] = data
        return data

    async def get_random_hit(
        self,
        content_type: Optional[str] = None,
        excluded_ids: Optional[Set[int]] = None,
    ) -> Optional[Dict]:
        """Returns a universally recognized, legendary movie or series from the Kinopoisk Top 250."""
        # Decide collection type: movies or tv shows
        if not content_type or content_type == "any":
            collection_type = random.choice(["TOP_250_MOVIES", "TOP_250_TV_SHOWS"])
        elif content_type == ContentType.TV.value:
            collection_type = "TOP_250_TV_SHOWS"
        else:
            collection_type = "TOP_250_MOVIES"

        pages_to_sample = random.sample(range(1, 14), 2)

        async with aiohttp.ClientSession() as session:
            candidates: List[Dict] = []

            for page in pages_to_sample:
                params = {"type": collection_type, "page": page}
                try:
                    headers = self._get_headers()
                    async with session.get(
                        Config.COLLECTIONS_URL,
                        headers=headers,
                        params=params,
                        timeout=Config.REQUEST_TIMEOUT,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            items = data.get("items", [])
                            for item in items:
                                f_id = item.get("kinopoiskId")
                                if f_id and (not excluded_ids or f_id not in excluded_ids):
                                    candidates.append(item)
                        elif resp.status == 402:
                            self._mark_key_exhausted()
                except KinopoiskQuotaExceededError:
                    raise
                except Exception as e:
                    logger.warning(f"Error fetching collection {collection_type} page {page}: {e}")

                if candidates:
                    break

            if not candidates:
                # Fallback to page 1
                try:
                    headers = self._get_headers()
                    params = {"type": collection_type, "page": 1}
                    async with session.get(
                        Config.COLLECTIONS_URL,
                        headers=headers,
                        params=params,
                        timeout=Config.REQUEST_TIMEOUT,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            candidates = [
                                i for i in data.get("items", [])
                                if not excluded_ids or i.get("kinopoiskId") not in excluded_ids
                            ]
                        elif resp.status == 402:
                            self._mark_key_exhausted()
                except KinopoiskQuotaExceededError:
                    raise
                except Exception:
                    pass

            if not candidates:
                return None

            random.shuffle(candidates)

            # Try up to 5 candidates to find one with complete details
            for candidate in candidates[:5]:
                film_id = candidate.get("kinopoiskId")
                if not film_id:
                    continue

                details = await self.get_film_details(film_id, session)
                if not details:
                    continue

                desc = details.get("description") or details.get("shortDescription")
                if not desc or len(desc.strip()) < 20:
                    continue

                # Ensure not documentary or other excluded genre in full details
                det_genres = {g.get("genre", "").lower() for g in details.get("genres", []) if "genre" in g}
                if any(ex in det_genres for ex in Config.EXCLUDED_GENRES):
                    continue

                poster = details.get("posterUrlPreview") or details.get("posterUrl")
                if not poster or "placeholder" in poster.lower():
                    continue

                return details

        return None
