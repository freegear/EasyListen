from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _json_dump(value: Any, ensure_ascii: bool = True) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    )


class RecordingStorage:
    def __init__(self, database_path: Path, recordings_dir: Path) -> None:
        self.database_path = database_path
        self.recordings_dir = recordings_dir
        self.lock = Lock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    duration REAL NOT NULL,
                    segments_json TEXT NOT NULL,
                    waveform_json TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    enhanced_audio_path TEXT,
                    diagnostics_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(recordings)")
            }
            if "diagnostics_json" not in columns:
                connection.execute(
                    "ALTER TABLE recordings "
                    "ADD COLUMN diagnostics_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "enhanced_audio_path" not in columns:
                connection.execute(
                    "ALTER TABLE recordings ADD COLUMN enhanced_audio_path TEXT"
                )
            for row in connection.execute(
                "SELECT id, segments_json, waveform_json, diagnostics_json "
                "FROM recordings"
            ).fetchall():
                connection.execute(
                    """
                    UPDATE recordings
                    SET segments_json = ?, waveform_json = ?, diagnostics_json = ?
                    WHERE id = ?
                    """,
                    (
                        _json_dump(
                            json.loads(row["segments_json"]),
                            ensure_ascii=False,
                        ),
                        _json_dump(json.loads(row["waveform_json"])),
                        _json_dump(
                            json.loads(row["diagnostics_json"]),
                            ensure_ascii=False,
                        ),
                        row["id"],
                    ),
                )

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "createdAt": row["created_at"],
            "duration": row["duration"],
            "segments": json_safe(json.loads(row["segments_json"])),
            "waveform": json_safe(json.loads(row["waveform_json"])),
            "diagnostics": json_safe(json.loads(row["diagnostics_json"])),
            "hasAudio": True,
            "hasEnhancedAudio": bool(row["enhanced_audio_path"]),
        }

    def save(
        self,
        recording: dict[str, Any],
        wav_bytes: bytes,
        enhanced_wav_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        recording = json_safe(recording)
        audio_path = self.recordings_dir / f"{recording['id']}.wav"
        audio_path.write_bytes(wav_bytes)
        enhanced_audio_path: Path | None = None
        if enhanced_wav_bytes is not None:
            enhanced_audio_path = (
                self.recordings_dir / f"{recording['id']}.enhanced.wav"
            )
            enhanced_audio_path.write_bytes(enhanced_wav_bytes)
        recording["hasEnhancedAudio"] = enhanced_audio_path is not None
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO recordings
                (id, title, created_at, duration, segments_json, waveform_json,
                 audio_path, enhanced_audio_path, diagnostics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recording["id"],
                    recording["title"],
                    recording["createdAt"],
                    recording["duration"],
                    _json_dump(recording["segments"], ensure_ascii=False),
                    _json_dump(recording["waveform"]),
                    str(audio_path),
                    str(enhanced_audio_path) if enhanced_audio_path else None,
                    _json_dump(
                        recording.get("diagnostics", {}),
                        ensure_ascii=False,
                    ),
                ),
            )
        return recording

    def list(self) -> list[dict[str, Any]]:
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM recordings ORDER BY created_at DESC"
            ).fetchall()
        return [self._serialize(row) for row in rows]

    def count(self) -> int:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS recording_count FROM recordings"
            ).fetchone()
        return int(row["recording_count"])

    def get(self, recording_id: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM recordings WHERE id = ?", (recording_id,)
            ).fetchone()
        return self._serialize(row) if row else None

    def update_segments(
        self,
        recording_id: str,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE recordings SET segments_json = ? WHERE id = ?",
                (
                    _json_dump(segments, ensure_ascii=False),
                    recording_id,
                ),
            )
        return self.get(recording_id) if cursor.rowcount else None

    def audio_path(
        self,
        recording_id: str,
        enhanced: bool = False,
    ) -> Path | None:
        column = "enhanced_audio_path" if enhanced else "audio_path"
        with self.lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT {column} AS requested_path FROM recordings WHERE id = ?",
                (recording_id,),
            ).fetchone()
        if not row or not row["requested_path"]:
            return None
        path = Path(row["requested_path"])
        return path if path.exists() else None

    def delete(self, recording_id: str) -> bool:
        audio_path = self.audio_path(recording_id)
        enhanced_audio_path = self.audio_path(recording_id, enhanced=True)
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM recordings WHERE id = ?", (recording_id,)
            )
        if audio_path:
            audio_path.unlink(missing_ok=True)
        if enhanced_audio_path:
            enhanced_audio_path.unlink(missing_ok=True)
        return cursor.rowcount > 0

    def clear(self) -> int:
        recordings = self.list()
        for recording in recordings:
            self.delete(recording["id"])
        return len(recordings)

    def cleanup_older_than(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat(
            timespec="seconds"
        )
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM recordings WHERE created_at < ?", (cutoff,)
            ).fetchall()
        for row in rows:
            self.delete(row["id"])
        return len(rows)
