from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(30), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    # relationships
    game_sessions = db.relationship("GameSession", back_populates="user", lazy="dynamic")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class GameSession(db.Model):
    __tablename__ = "game_sessions"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    mode         = db.Column(db.String(20), nullable=False)   # free | multiple_choice
    timed        = db.Column(db.Boolean, default=False)
    multiplayer  = db.Column(db.Boolean, default=False)
    rounds       = db.Column(db.Integer, nullable=False)
    final_mu     = db.Column(db.Float, nullable=False)
    accuracy     = db.Column(db.Float, nullable=False)        # 0.0 – 1.0
    best_streak  = db.Column(db.Integer, default=0)
    played_at    = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # relationships
    user    = db.relationship("User", back_populates="game_sessions")
    records = db.relationship("QuestionRecord", back_populates="session",
                              cascade="all, delete-orphan", lazy="dynamic")

    @property
    def accuracy_pct(self):
        return round(self.accuracy * 100, 1)

    def __repr__(self):
        return f"<GameSession user={self.user_id} mu={self.final_mu:.2f}>"


class QuestionRecord(db.Model):
    """One row per question answered within a session."""
    __tablename__ = "question_records"

    id            = db.Column(db.Integer, primary_key=True)
    session_id    = db.Column(db.Integer, db.ForeignKey("game_sessions.id"),
                              nullable=False, index=True)
    question_id   = db.Column(db.Integer, nullable=False)
    category      = db.Column(db.String(40))
    difficulty    = db.Column(db.Float)
    is_correct    = db.Column(db.Boolean, nullable=False)
    response_time = db.Column(db.Float)
    mu_before     = db.Column(db.Float)
    mu_after      = db.Column(db.Float)
    hint_used     = db.Column(db.Boolean, default=False)
    player        = db.Column(db.Integer, default=1)          # 1 or 2

    session = db.relationship("GameSession", back_populates="records")

    def __repr__(self):
        return f"<QRecord q={self.question_id} correct={self.is_correct}>"