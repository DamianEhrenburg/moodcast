import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "moodcast.db"


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS watchlist (
                        user_id INTEGER NOT NULL,
                        film_id INTEGER NOT NULL,
                        name_ru TEXT NOT NULL,
                        name_original TEXT,
                        year TEXT,
                        rating TEXT,
                        genres TEXT,
                        poster_url TEXT,
                        web_url TEXT,
                        is_watched INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, film_id)
                    )
                    """
                )
                conn.commit()
            logger.info("Watchlist database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)

    def add_to_watchlist(self, user_id: int, film: Dict) -> bool:
        film_id = film.get("kinopoiskId")
        if not film_id:
            return False

        name_ru = film.get("nameRu") or film.get("nameOriginal") or "Без названия"
        name_orig = film.get("nameOriginal") or ""
        year = str(film.get("year") or "")
        rating = str(film.get("ratingKinopoisk") or film.get("rating") or "N/A")

        genres_list = [g.get("genre", "") for g in film.get("genres", []) if "genre" in g]
        genres = ", ".join(genres_list)

        poster_url = film.get("posterUrlPreview") or film.get("posterUrl") or ""
        web_url = film.get("webUrl") or f"https://www.kinopoisk.ru/film/{film_id}/"

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO watchlist (
                        user_id, film_id, name_ru, name_original, year, rating,
                        genres, poster_url, web_url, is_watched
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(user_id, film_id) DO NOTHING
                    """,
                    (user_id, film_id, name_ru, name_orig, year, rating, genres, poster_url, web_url),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding film {film_id} to watchlist: {e}")
            return False

    def remove_from_watchlist(self, user_id: int, film_id: int) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM watchlist WHERE user_id = ? AND film_id = ?",
                    (user_id, film_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error removing film {film_id} from watchlist: {e}")
            return False

    def is_in_watchlist(self, user_id: int, film_id: int) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM watchlist WHERE user_id = ? AND film_id = ?",
                    (user_id, film_id),
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking watchlist for {film_id}: {e}")
            return False

    def get_user_watchlist(
        self, user_id: int, page: int = 1, page_size: int = 5
    ) -> Tuple[List[Dict], int]:
        offset = (page - 1) * page_size
        try:
            with self._get_connection() as conn:
                total_cursor = conn.execute(
                    "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
                )
                total_count = total_cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT * FROM watchlist
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (user_id, page_size, offset),
                )
                rows = [dict(row) for row in cursor.fetchall()]
            return rows, total_count
        except Exception as e:
            logger.error(f"Error fetching watchlist for user {user_id}: {e}")
            return [], 0

    def toggle_watched(self, user_id: int, film_id: int) -> Optional[bool]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT is_watched FROM watchlist WHERE user_id = ? AND film_id = ?",
                    (user_id, film_id),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                new_state = 0 if row["is_watched"] == 1 else 1
                conn.execute(
                    "UPDATE watchlist SET is_watched = ? WHERE user_id = ? AND film_id = ?",
                    (new_state, user_id, film_id),
                )
                conn.commit()
                return bool(new_state)
        except Exception as e:
            logger.error(f"Error toggling watched status: {e}")
            return None


db = Database()
