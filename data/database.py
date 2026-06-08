# database.py — Sample database helper for CodeRAG testing

import sqlite3
from typing import Optional


DB_PATH = "app.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open and return a SQLite database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the users and products tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            price    REAL NOT NULL,
            stock    INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def insert_user(conn: sqlite3.Connection,
                username: str, email: str, password_hash: str) -> int:
    """Insert a new user. Returns the new row ID."""
    cursor = conn.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (username, email, password_hash)
    )
    conn.commit()
    return cursor.lastrowid


def get_user_by_username(conn: sqlite3.Connection,
                          username: str) -> Optional[dict]:
    """Fetch a user row by username. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    return dict(row) if row else None


def update_stock(conn: sqlite3.Connection,
                 product_id: int, quantity: int) -> bool:
    """
    Reduce product stock by quantity.
    Returns False if stock would go negative.
    """
    current = conn.execute(
        "SELECT stock FROM products WHERE id = ?", (product_id,)
    ).fetchone()

    if not current or current["stock"] < quantity:
        return False

    conn.execute(
        "UPDATE products SET stock = stock - ? WHERE id = ?",
        (quantity, product_id)
    )
    conn.commit()
    return True


def search_products(conn: sqlite3.Connection, keyword: str) -> list:
    """Search products by name keyword. Returns list of matching rows."""
    rows = conn.execute(
        "SELECT * FROM products WHERE name LIKE ?",
        (f"%{keyword}%",)
    ).fetchall()
    return [dict(r) for r in rows]
