import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import requests
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, GameAction, GameState

from .agent import Agent


class AegeanAgent(Agent):
    """
    ARC agent powered by aegean-consensus via HTTP API.

    Flow per step:
      1) Build an ARC state prompt from latest frame
      2) Call aegean-consensus group consensus endpoint
      3) Parse final answer into ARC action (RESET/ACTION1..ACTION6)
      4) Fallback safely if malformed output

    Extra telemetry:
      - per-game token totals
      - per-game consensus latency totals
      - optional local CSV export
    """

    MAX_ACTIONS = 80

    _csv_lock = Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = os.getenv("AEGEAN_BASE_URL", "http://localhost:8000").rstrip("/")
        self.timeout = int(os.getenv("AEGEAN_TIMEOUT_SEC", "10"))
        self.max_rounds = int(os.getenv("AEGEAN_ARC_MAX_ROUNDS", "1"))
        self.quorum_threshold = float(os.getenv("AEGEAN_ARC_QUORUM", "0.5"))
        self.agent_count = int(os.getenv("AEGEAN_ARC_AGENT_COUNT", "3"))
        self.include_frame = os.getenv("AEGEAN_ARC_INCLUDE_FRAME", "true").lower() == "true"
        self.max_prompt_chars = int(os.getenv("AEGEAN_ARC_MAX_PROMPT_CHARS", "12000"))

        self.group_id: Optional[str] = None
        self._group_deleted = False

        # Per-game telemetry
        self.step_count = 0
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.total_consensus_latency_s = 0.0
        self.last_raw_answer = ""

    @property
    def name(self) -> str:
        return f"{super().name}.aegean"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            self._ensure_group()

            task = self._build_task(frames, latest_frame)
            t0 = time.perf_counter()
            resp = requests.post(
                f"{self.base_url}/api/v1/groups/{self.group_id}/consensus",
                json={
                    "task": task,
                    "quorum_threshold": self.quorum_threshold,
                    "max_rounds": self.max_rounds,
                },
                timeout=self.timeout,
            )
            latency = time.perf_counter() - t0
            self.total_consensus_latency_s += latency
            self.step_count += 1

            resp.raise_for_status()
            data = resp.json()

            tp, tc = self._extract_tokens(data)
            self.total_tokens_prompt += tp
            self.total_tokens_completion += tc

            final_solution = data.get("final_solution") or {}
            raw_answer = str(final_solution.get("answer", "")).strip()
            self.last_raw_answer = raw_answer

            action = self._safe_parse_action(raw_answer, latest_frame)
            if action.is_simple():
                action.reasoning = (
                    f"aegean-consensus action={action.name} "
                    f"tokens={tp + tc} latency={latency:.3f}s"
                )
            return action

        except Exception as e:  # noqa: BLE001
            return self._fallback_action(latest_frame, f"aegean_error: {e}")

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        # Ensure group is removed at game end
        self._delete_group_if_needed()

        # Export game-level metrics to local CSV
        try:
            self._append_game_metrics_csv()
        except Exception:
            pass

        super().cleanup(scorecard)

    def _ensure_group(self) -> None:
        if self.group_id:
            return

        group_resp = requests.post(
            f"{self.base_url}/api/v1/groups/",
            json={
                "group_name": f"arc-{self.game_id}",
                "mode": "consensus",
                "created_by": "arc-agent",
            },
            timeout=self.timeout,
        )
        group_resp.raise_for_status()
        self.group_id = group_resp.json()["group_id"]

        agents_resp = requests.get(
            f"{self.base_url}/api/v1/groups/agents",
            timeout=self.timeout,
        )
        agents_resp.raise_for_status()
        available_agents = agents_resp.json()
        if not available_agents:
            raise RuntimeError("No registered agents available in aegean-consensus")

        target_count = min(self.agent_count, len(available_agents))
        if target_count <= 0:
            raise RuntimeError("No agents can be added to aegean group")

        for i in range(target_count):
            agent_id = available_agents[i]["agent_id"]
            add_resp = requests.post(
                f"{self.base_url}/api/v1/groups/{self.group_id}/members",
                json={
                    "agent_id": agent_id,
                    "role": "arc_player",
                    "capability_weight": 1.0,
                },
                timeout=self.timeout,
            )
            add_resp.raise_for_status()

    def _delete_group_if_needed(self) -> None:
        if self._group_deleted or not self.group_id:
            return
        try:
            requests.delete(
                f"{self.base_url}/api/v1/groups/{self.group_id}",
                timeout=self.timeout,
            )
        finally:
            self._group_deleted = True

    def _build_task(self, frames: list[FrameData], latest_frame: FrameData) -> str:
        frame_payload = ""
        if self.include_frame:
            try:
                frame_payload = json.dumps(latest_frame.frame, separators=(",", ":"))
            except Exception:
                frame_payload = str(latest_frame.frame)

        available_actions = []
        try:
            available_actions = [a.name for a in latest_frame.available_actions]
        except Exception:
            pass

        prompt = (
            "You are controlling an ARC-AGI-3 environment.\n"
            "Return EXACTLY one valid action token and nothing else:\n"
            "RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6\n\n"
            f"state={latest_frame.state.name}\n"
            f"levels_completed={latest_frame.levels_completed}\n"
            f"available_actions={available_actions}\n"
        )

        if frame_payload:
            prompt += f"frame={frame_payload}\n"

        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[: self.max_prompt_chars] + "\n[TRUNCATED]"

        return prompt

    def _safe_parse_action(self, text: str, latest_frame: FrameData) -> GameAction:
        if not text:
            return self._fallback_action(latest_frame, "empty answer")

        # Accept direct token, JSON-like strings, or verbose output
        upper = text.upper().strip()
        m = re.search(r"(RESET|ACTION[1-6])", upper)
        if not m:
            return self._fallback_action(latest_frame, f"invalid answer: {text[:80]}")

        name = m.group(1)
        action = GameAction.from_name(name)

        # ACTION6 requires coordinates; provide safe center default
        if action is GameAction.ACTION6:
            action.set_data({"x": 32, "y": 32})

        return action

    def _fallback_action(self, latest_frame: FrameData, reason: str) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            # Safe fallback: avoid ACTION6 when no coordinates are available
            action = GameAction.ACTION5

        if action.is_simple():
            action.reasoning = f"fallback: {reason}"
        return action

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _extract_tokens(self, payload: dict[str, Any]) -> tuple[int, int]:
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            usage = {}

        prompt = max(
            self._to_int(payload.get("tokens_prompt") if isinstance(payload, dict) else 0),
            self._to_int(usage.get("tokens_prompt")),
            self._to_int(usage.get("prompt_tokens")),
            self._to_int(usage.get("input_tokens")),
        )
        completion = max(
            self._to_int(payload.get("tokens_completion") if isinstance(payload, dict) else 0),
            self._to_int(usage.get("tokens_completion")),
            self._to_int(usage.get("completion_tokens")),
            self._to_int(usage.get("output_tokens")),
        )

        total = max(
            self._to_int(usage.get("tokens_total")),
            self._to_int(usage.get("total_tokens")),
        )
        if total > 0 and prompt + completion == 0:
            prompt = int(total * 0.8)
            completion = total - prompt

        return prompt, completion

    def _append_game_metrics_csv(self) -> None:
        csv_path = Path(os.getenv("AEGEAN_ARC_METRICS_CSV", "aegean_arc_metrics.csv"))
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        total_tokens = self.total_tokens_prompt + self.total_tokens_completion
        avg_latency = self.total_consensus_latency_s / self.step_count if self.step_count else 0.0

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "game_id": self.game_id,
            "agent_name": self.agent_name,
            "card_id": self.card_id,
            "group_id": self.group_id or "",
            "steps": self.step_count,
            "tokens_prompt": self.total_tokens_prompt,
            "tokens_completion": self.total_tokens_completion,
            "tokens_total": total_tokens,
            "consensus_latency_total_s": round(self.total_consensus_latency_s, 4),
            "consensus_latency_avg_s": round(avg_latency, 4),
            "levels_completed": self.levels_completed,
            "final_state": self.state.name,
            "last_raw_answer": self.last_raw_answer[:120],
        }

        fieldnames = list(row.keys())
        with self._csv_lock:
            file_exists = csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

