import datetime

from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    JSON,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .db import Base

load_dotenv()


class App(Base):
    __tablename__ = "apps"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True, unique=True)
    users = relationship("User", back_populates="app")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
    )


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)
    app = relationship("App", back_populates="users")
    sessions = relationship("Session", back_populates="user")
    collections = relationship("Collection", back_populates="user")
    metamessages = relationship("Metamessage", back_populates="user")

    __table_args__ = (
        UniqueConstraint("name", "app_id", name="unique_name_app_user"),
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        Index("idx_users_app_lookup", "app_id", "public_id"),
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, app_id={self.app_id}, public_id={self.public_id} created_at={self.created_at}, h_metadata={self.h_metadata})"


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    messages = relationship("Message", back_populates="session")
    metamessages = relationship("Metamessage", back_populates="session")
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        Index("idx_sessions_user_lookup", "user_id", "public_id"),
    )

    def __repr__(self) -> str:
        return f"Session(id={self.id}, user_id={self.user_id}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.public_id"), index=True
    )
    is_user: Mapped[bool]
    content: Mapped[str] = mapped_column(TEXT)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    session = relationship("Session", back_populates="messages")
    metamessages = relationship("Metamessage", back_populates="message")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        Index(
            "idx_messages_session_lookup",
            "session_id",
            "id",
            postgresql_include=["public_id", "is_user", "created_at"],
        ),
    )

    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_id={self.session_id}, is_user={self.is_user}, content={self.content[10:]})"


class Metamessage(Base):
    __tablename__ = "metamessages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    metamessage_type: Mapped[str] = mapped_column(TEXT, index=True)
    content: Mapped[str] = mapped_column(TEXT)

    # Foreign keys - message_id is now optional
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.public_id"), index=True, nullable=True
    )
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.public_id"), index=True, nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="metamessages")
    session = relationship("Session", back_populates="metamessages")
    message = relationship("Message", back_populates="metamessages")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint(
            "length(metamessage_type) <= 512", name="metamessage_type_length"
        ),
        # Added constraints to ensure consistency
        CheckConstraint(
            "(message_id IS NULL) OR (session_id IS NOT NULL)",
            name="message_requires_session",
        ),
        # Keep existing index
        Index(
            "idx_metamessages_lookup",
            "metamessage_type",
            text("id DESC"),
            postgresql_include=["public_id", "message_id", "created_at"],
        ),
        # Indices for user, session, and message lookups
        Index(
            "idx_metamessages_user_lookup",
            "user_id",
            "metamessage_type",
            text("id DESC"),
        ),
        Index(
            "idx_metamessages_session_lookup",
            "session_id",
            "metamessage_type",
            text("id DESC"),
        ),
        Index(
            "idx_metamessages_message_lookup",
            "message_id",
            "metamessage_type",
            text("id DESC"),
        ),
    )

    def __repr__(self) -> str:
        return f"Metamessages(id={self.id}, user_id={self.user_id}, session_id={self.session_id}, message_id={self.message_id}, metamessage_type={self.metamessage_type})"


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete, delete-orphan"
    )
    user = relationship("User", back_populates="collections")
    user_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("users.public_id"), index=True
    )

    __table_args__ = (
        UniqueConstraint("name", "user_id", name="unique_name_collection_user"),
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(name) <= 512", name="name_length"),
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    content: Mapped[str] = mapped_column(TEXT)
    embedding = mapped_column(Vector(1536))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )

    collection_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("collections.public_id"), index=True
    )
    collection = relationship("Collection", back_populates="documents")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
    )


class QueueItem(Base):
    __tablename__ = "queue"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)


class ActiveQueueSession(Base):
    __tablename__ = "active_queue_sessions"

    session_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    last_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )


class Demographics(Base):
    """Represents demographic information of a user."""
    __tablename__ = "demographics"
    id = Column(String, primary_key=True, index=True)
    gender = Column(String, nullable=False)  # Gender of the user (e.g., male, female, non_binary, decline_to_state)
    age_range = Column(String, nullable=False)  # Age range of the user (e.g., <18, 18-25, 26-40, 41-60, 60+)


class HabitsPermissions(Base):
    """Represents user habits and permissions for data collection."""
    __tablename__ = "habits_permissions"
    id = Column(String, primary_key=True, index=True)
    bedtime = Column(String, nullable=False)  # User's bedtime habit (e.g., before_22:00, around_23:00, after_midnight, irregular)
    biometric_consent = Column(Boolean, nullable=False)  # Whether the user consents to biometric data collection (true/false)


class EmotionalState(Base):
    """Represents the primary emotions of a user."""
    __tablename__ = "emotional_state"
    id = Column(String, primary_key=True, index=True)
    primary_emotions = Column(JSONB, nullable=False)  # List of user's recent primary emotions (e.g., anxiety, happiness, calmness)


class MBTIPersonality(Base):
    """Represents the MBTI personality assessment of a user."""
    __tablename__ = "mbti_personality"
    id = Column(String, primary_key=True, index=True)
    energy = Column(String, nullable=False)  # User's energy type (e.g., extrovert, introvert)
    information = Column(String, nullable=False)  # User's information processing style (e.g., sensing, intuitive)
    decision = Column(String, nullable=False)  # User's decision-making style (e.g., thinking, feeling)
    lifestyle = Column(String, nullable=False)  # User's lifestyle preference (e.g., judging, perceiving)


class StressLevel(Base):
    """Represents the stress level of a user."""
    __tablename__ = "stress_level"
    id = Column(String, primary_key=True, index=True)
    stress_level = Column(String, nullable=False)  # User's stress level (e.g., 0.0-0.3: Beginner Sanctuary, >0.9: Endgame Warfare)


class UserPersona(Base):
    """Represents the complete persona of a user."""
    __tablename__ = "user_personas"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)  # Foreign key to users table
    demographics_id = Column(String, ForeignKey("demographics.id"), nullable=False)  # Foreign key to demographics table
    habits_permissions_id = Column(String, ForeignKey("habits_permissions.id"), nullable=False)  # Foreign key to habits_permissions table
    emotional_state_id = Column(String, ForeignKey("emotional_state.id"), nullable=False)  # Foreign key to emotional_state table
    personality_assessment_id = Column(String, ForeignKey("mbti_personality.id"), nullable=False)  # Foreign key to mbti_personality table
    stress_assessment_id = Column(String, ForeignKey("stress_level.id"), nullable=False)  # Foreign key to stress_level table

    demographics = relationship("Demographics")
    habits_permissions = relationship("HabitsPermissions")
    emotional_state = relationship("EmotionalState")
    personality_assessment = relationship("MBTIPersonality")
    stress_assessment = relationship("StressLevel")
