# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Persistent SQLite logger for training runs.

Schema:
    runs        — one row per training run
    steps       — one row per optimizer step (loss, reward, kl, lr, ...)
    completions — one row per completion (prompt, text, predicted, expected, reward)
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("runs.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,
    trainer    TEXT NOT NULL,
    config     TEXT NOT NULL,
    duration   TEXT,
    status     TEXT NOT NULL DEFAULT 'running'
);
CREATE TABLE IF NOT EXISTS steps (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT NOT NULL,
    step                  INTEGER NOT NULL,
    timestamp             TEXT NOT NULL,
    loss                  REAL,
    pg_loss               REAL,
    reward                REAL,
    kl                    REAL,
    lr                    REAL,
    truncated_count       INTEGER,
    grad_norm             REAL,
    format_rate           REAL,
    avg_completion_length REAL,
    step_time             REAL,
    gen_time              REAL
);
CREATE TABLE IF NOT EXISTS completions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    step             INTEGER NOT NULL,
    prompt           TEXT,
    completion       TEXT,
    predicted_answer TEXT,
    expected_answer  TEXT,
    reward           REAL,
    advantage        REAL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step);
CREATE INDEX IF NOT EXISTS idx_comps_run ON completions(run_id, step);
"""


class RunLogger:
    """One instance per training run. Single-threaded, persistent connection."""

    def __init__(self, trainer: str, config: dict, duration: str | None = None):
        self._con = sqlite3.connect(DB_PATH, timeout=30)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.executescript(_SCHEMA)

        ts = datetime.now(timezone.utc)
        self.run_id = ts.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._con.execute(
            "INSERT INTO runs (run_id, start_time, trainer, config, duration) VALUES (?,?,?,?,?)",
            (self.run_id, ts.isoformat(), trainer, json.dumps(config), duration),
        )
        self._con.commit()
        print(f"[RunLogger] run_id={self.run_id}  db={DB_PATH.resolve()}")

    def log_step(
        self,
        step: int,
        *,
        loss: float | None = None,
        pg_loss: float | None = None,
        reward: float | None = None,
        kl: float | None = None,
        lr: float | None = None,
        truncated_count: int | None = None,
        grad_norm: float | None = None,
        format_rate: float | None = None,
        avg_completion_length: float | None = None,
        step_time: float | None = None,
        gen_time: float | None = None,
    ):
        self._con.execute(
            "INSERT INTO steps"
            " (run_id, step, timestamp, loss, pg_loss, reward, kl, lr,"
            "  truncated_count, grad_norm, format_rate, avg_completion_length,"
            "  step_time, gen_time)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id,
                step,
                datetime.now(timezone.utc).isoformat(),
                loss,
                pg_loss,
                reward,
                kl,
                lr,
                truncated_count,
                grad_norm,
                format_rate,
                avg_completion_length,
                step_time,
                gen_time,
            ),
        )
        self._con.commit()

    def log_completions(
        self,
        step: int,
        *,
        prompts: list,
        completions: list,
        predicted: list,
        expected: list,
        rewards: list,
        advantages: list | None = None,
    ):
        if advantages is None:
            advantages = [None] * len(completions)
        self._con.executemany(
            "INSERT INTO completions"
            " (run_id, step, prompt, completion, predicted_answer,"
            "  expected_answer, reward, advantage)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [
                (self.run_id, step, p, c, pred, exp, rew, adv)
                for p, c, pred, exp, rew, adv in zip(
                    prompts, completions, predicted, expected, rewards, advantages
                )
            ],
        )
        self._con.commit()

    def finish(self, status: str = "done"):
        self._con.execute("UPDATE runs SET status=? WHERE run_id=?", (status, self.run_id))
        self._con.commit()
        self._con.close()
