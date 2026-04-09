# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Web viewer for browsing logged training runs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from flask import Flask, abort, g, render_template

DEFAULT_DB_PATH = Path("runs.db")


def create_app(db_path: Path | str = DEFAULT_DB_PATH) -> Flask:
    """Create the Flask application bound to a specific SQLite database."""

    app = Flask(__name__)
    app.config["RUNS_DB_PATH"] = Path(db_path)

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            db_path = app.config["RUNS_DB_PATH"]
            if not db_path.exists():
                abort(503, description=f"Database not found: {db_path}. Run training first.")
            g.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_error: BaseException | None = None) -> None:
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.route("/")
    def runs():
        db = get_db()
        rows = db.execute(
            """
            SELECT r.*,
                   COUNT(s.step) AS step_count,
                   ROUND(AVG(s.reward), 4) AS avg_reward,
                   MAX(s.step) AS last_step,
                   (
                       SELECT COUNT(*)
                       FROM completions c
                       WHERE c.run_id = r.run_id
                   ) AS completion_count
            FROM runs r
            LEFT JOIN steps s ON s.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.start_time DESC
            """
        ).fetchall()
        return render_template("runs.html", runs=rows)

    @app.route("/run/<run_id>")
    def run_detail(run_id: str):
        db = get_db()
        run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            abort(404)
        steps = db.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY step",
            (run_id,),
        ).fetchall()
        config = json.loads(run["config"])
        total_completions = _total_completions_per_step(config)
        return render_template(
            "run.html",
            run=run,
            steps=steps,
            config=config,
            total_completions_per_step=total_completions,
        )

    @app.route("/run/<run_id>/step/<int:step>")
    def step_detail(run_id: str, step: int):
        db = get_db()
        run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            abort(404)

        rows = db.execute(
            "SELECT * FROM completions WHERE run_id=? AND step=? ORDER BY id",
            (run_id, step),
        ).fetchall()

        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            prompt = row["prompt"] or ""
            groups.setdefault(prompt, []).append(row)

        step_stats = db.execute(
            "SELECT * FROM steps WHERE run_id=? AND step=?",
            (run_id, step),
        ).fetchone()
        prev_step = db.execute(
            "SELECT MAX(step) FROM steps WHERE run_id=? AND step<?",
            (run_id, step),
        ).fetchone()[0]
        next_step = db.execute(
            "SELECT MIN(step) FROM steps WHERE run_id=? AND step>?",
            (run_id, step),
        ).fetchone()[0]

        return render_template(
            "step.html",
            run=run,
            step=step,
            step_stats=step_stats,
            groups=groups,
            prev_step=prev_step,
            next_step=next_step,
        )

    return app


def _total_completions_per_step(config: dict) -> int | None:
    num_generations = config.get("num_generations")
    prompts_per_step = config.get("prompts_per_step") or 1
    if num_generations is None:
        return None
    return num_generations * prompts_per_step


def main() -> None:
    """Run the local viewer development server."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Note: {args.db} not found. Run training first to generate data.")

    app = create_app(args.db)
    print(f"Starting viewer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
