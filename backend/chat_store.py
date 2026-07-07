"""
聊天持久化存储 (Chat Store)

SQLite 数据库存储会话和消息历史。
支持创建/切换会话、保存消息、加载历史。

Schema:
    sessions:
        id TEXT PRIMARY KEY
        title TEXT
        created_at TEXT
        updated_at TEXT

    messages:
        id INTEGER PRIMARY KEY AUTOINCREMENT
        session_id TEXT (FK -> sessions.id)
        role TEXT (user/assistant/system)
        content TEXT
        feedback TEXT (NULL/like/dislike)
        created_at TEXT

    feedback:
        id INTEGER PRIMARY KEY AUTOINCREMENT
        message_id INTEGER (FK -> messages.id)
        rating TEXT (like/dislike)
        comment TEXT
        created_at TEXT
"""

import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional


DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "firefly_chat.db"


def get_db_path() -> str:
    """获取数据库路径"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return str(DB_PATH)


def init_db():
    """初始化数据库表"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '新对话',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            feedback TEXT DEFAULT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            rating TEXT NOT NULL CHECK(rating IN ('like', 'dislike')),
            comment TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions(updated_at DESC);
    """)

    conn.commit()
    conn.close()


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class ChatStore:
    """聊天持久化存储"""

    @staticmethod
    def create_session(title: str = "新对话") -> dict:
        """创建新会话"""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        conn = get_connection()
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        conn.commit()
        conn.close()

        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def list_sessions(limit: int = 50) -> list:
        """列出所有会话（按更新时间倒序）"""
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()

        return [dict(r) for r in rows]

    @staticmethod
    def get_session(session_id: str) -> Optional[dict]:
        """获取单个会话信息"""
        conn = get_connection()
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        conn.close()

        return dict(row) if row else None

    @staticmethod
    def delete_session(session_id: str):
        """删除会话（级联删除消息）"""
        conn = get_connection()
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()

    @staticmethod
    def update_session_title(session_id: str, title: str):
        """更新会话标题"""
        now = datetime.now().isoformat()
        conn = get_connection()
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def add_message(session_id: str, role: str, content: str) -> dict:
        """添加消息"""
        now = datetime.now().isoformat()

        conn = get_connection()
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        message_id = cursor.lastrowid

        # 更新会话时间
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )

        # 自动更新标题（使用第一条用户消息）
        if role == "user":
            existing = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ? AND role = 'user'",
                (session_id,),
            ).fetchone()
            if existing and existing["cnt"] == 1:
                title = content[:30] + ("..." if len(content) > 30 else "")
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )

        conn.commit()
        conn.close()

        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    @staticmethod
    def get_messages(session_id: str, limit: int = 100) -> list:
        """获取会话的所有消息"""
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, session_id, role, content, feedback, created_at "
            "FROM messages WHERE session_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()

        return [dict(r) for r in rows]

    @staticmethod
    def record_feedback(message_id: int, rating: str, comment: str = ""):
        """记录用户反馈"""
        conn = get_connection()

        # 更新消息的 feedback 字段
        conn.execute(
            "UPDATE messages SET feedback = ? WHERE id = ?",
            (rating, message_id),
        )

        # 插入详细反馈
        conn.execute(
            "INSERT INTO feedback (message_id, rating, comment) VALUES (?, ?, ?)",
            (message_id, rating, comment),
        )

        conn.commit()
        conn.close()

    @staticmethod
    def get_feedback_stats() -> dict:
        """获取反馈统计"""
        conn = get_connection()
        like = conn.execute(
            "SELECT COUNT(*) as cnt FROM feedback WHERE rating = 'like'"
        ).fetchone()
        dislike = conn.execute(
            "SELECT COUNT(*) as cnt FROM feedback WHERE rating = 'dislike'"
        ).fetchone()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE feedback IS NOT NULL"
        ).fetchone()
        conn.close()

        return {
            "total_rated": total["cnt"],
            "likes": like["cnt"],
            "dislikes": dislike["cnt"],
            "satisfaction_rate": round(
                like["cnt"] / max(1, total["cnt"]), 3
            ),
        }


# 初始化数据库（模块导入时自动执行）
init_db()
