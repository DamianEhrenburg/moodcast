import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ContentType(Enum):
    MOVIE = "movie"
    TV = "tv"


class EmotionType(Enum):
    EXCITED = auto()
    TENSE = auto()
    HAPPY = auto()
    CALM = auto()
    SAD = auto()
    BORED = auto()
    INSPIRED = auto()
    ROMANTIC = auto()
    THOUGHTFUL = auto()
    NEUTRAL = auto()
    HORROR = auto()
    SARCASTIC = auto()
    ANGRY = auto()
    CONFUSED = auto()
    SURPRISED = auto()


@dataclass
class EmotionalState:
    primary_emotion: str
    intensity: float = 0.0
    suggested_genres: List[str] = field(default_factory=list)
    excluded_genres: List[str] = field(default_factory=list)


@dataclass
class UserState:
    user_id: int
    type_selected: Optional[str] = None
    emotional_state: Optional[EmotionalState] = None
    recommendations: List[Dict] = field(default_factory=list)
    current_index: int = 0
    shown_recommendations: Set[int] = field(default_factory=set)
    shown_history: List[int] = field(default_factory=list)
    history_index: int = -1
    last_request_time: Optional[datetime] = None
    min_interval: int = 3
    message_history: List[str] = field(default_factory=list)
    max_history: int = 5
    is_blocked: bool = False
    last_message_id: Optional[int] = None
    last_feedback_message_id: Optional[int] = None

    def can_make_request(self) -> bool:
        if self.last_request_time is None:
            return True
        now = datetime.now()
        allowed = now - self.last_request_time >= timedelta(seconds=self.min_interval)
        if not allowed:
            logger.warning(
                f"User {self.user_id} rate limited: "
                f"{now - self.last_request_time} < {self.min_interval}s"
            )
        return allowed

    def update_request_time(self) -> None:
        self.last_request_time = datetime.now()

    def reset_recommendations(self) -> None:
        self.recommendations = []
        self.current_index = 0
        self.shown_history.clear()
        self.history_index = -1
        self.last_feedback_message_id = None
        self.emotional_state = None

    def add_shown_recommendation(self, film_id: int) -> None:
        self.shown_recommendations.add(film_id)
        current_idx = self.current_index

        is_new_step = (
            not self.shown_history
            or self.history_index < 0
            or self.shown_history[self.history_index] != current_idx
        )

        if is_new_step:
            self.history_index += 1
            self.shown_history = self.shown_history[:self.history_index]
            self.shown_history.append(current_idx)

    def is_recommendation_shown(self, film_id: int) -> bool:
        return film_id in self.shown_recommendations

    def add_message(self, message: str) -> None:
        if len(self.message_history) >= self.max_history:
            self.message_history.pop(0)
        self.message_history.append(message)


user_states: Dict[int, UserState] = {}
