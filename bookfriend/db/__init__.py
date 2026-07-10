"""
Database package exports.
"""

from .database import (
    Base,
    SessionLocal,
    engine,
    get_db,
    init_db,
)

from .repositories import (
    register_book,
    book_exists_by_filename,
    delete_book,
    create_user,
    user_exists,
    create_job,
    update_job,
    get_job,
    log_message,
    get_chat_history,
)

__all__ = [
    "Base",
    "SessionLocal",
    "engine",
    "get_db",
    "init_db",

    "register_book",
    "book_exists_by_filename",
    "delete_book",

    "create_user",
    "user_exists",

    "create_job",
    "update_job",
    "get_job",

    "log_message",
    "get_chat_history",
]