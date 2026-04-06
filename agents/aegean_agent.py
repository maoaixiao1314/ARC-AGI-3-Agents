import csv
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import requests
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, GameAction, GameState

from .agent import Agent

logger = logging.getLogger()


class AegeanAgent(Agent):
    """
    ARC agent powered by aegean-consensus via HTTP API.

    This version adds four practical safeguards:
      1) Reuses one Aegean group per ARC game
      2) Prevents RESET spam during active gameplay
      3) Sends a compact frame summary instead of full raw grid
      4) Logs enough debug context to inspect consensus behavior
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
        self.max_prompt_chars = int(os.getenv("AEGEAN_ARC_MAX_PROMPT_CHARS", "4000"))
        self.debug = os.getenv("AEGEAN_ARC_DEBUG", "true").lower() == "true"
        self.max_history = int(os.getenv("AEGEAN_ARC_HISTORY", "6"))

        self.group_id: Optional[str] = None
        self._group_deleted = False
        self._metrics_written = False

        # Per-game telemetry
        self.step_count = 0
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.total_consensus_latency_s = 0.0
        self.last_raw_answer = ""
        self.last_action_name = ""
        self.last_fallback_reason = ""

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
            self.last_action_name = action.name

            if action.is_simple():
                action.reasoning = (
                    f"aegean-consensus action={action.name} "
                    f"tokens={tp + tc} latency={latency:.3f}s"
                )

            self._debug_log(
                step=self.step_count,
                state=latest_frame.state.name,
                raw_answer=raw_answer,
                action=action.name,
                prompt_tokens=tp,
                completion_tokens=tc,
                latency=latency,
            )
            return action

        except Exception as e:  # noqa: BLE001
            action = self._fallback_action(latest_frame, f"aegean_error: {e}")
            self.last_action_name = action.name
            self._debug_log(
                step=self.step_count,
                state=latest_frame.state.name,
                raw_answer="",
                action=action.name,
                prompt_tokens=0,
                completion_tokens=0,
                latency=0.0,
                error=str(e),
            )
            return action

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        self._delete_group_if_needed()

        try:
            self._append_game_metrics_csv()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to write Aegean metrics CSV: %s", exc)

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
        logger.info("Aegean group created for %s: %s", self.game_id, self.group_id)

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

        selected_ids: list[str] = []
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
            selected_ids.append(agent_id)
        logger.info("Aegean group %s members: %s", self.group_id, selected_ids)

    def _delete_group_if_needed(self) -> None:
        if self._group_deleted or not self.group_id:
            return
        try:
            requests.delete(
                f"{self.base_url}/api/v1/groups/{self.group_id}",
                timeout=self.timeout,
            )
            logger.info("Aegean group deleted for %s: %s", self.game_id, self.group_id)
        finally:
            self._group_deleted = True

    def _build_task(self, frames: list[FrameData], latest_frame: FrameData) -> str:
        available_actions = self._available_action_names(latest_frame)
        frame_summary = self._summarize_frame(latest_frame) if self.include_frame else "frame_summary=disabled"
        recent_actions = self._recent_actions(frames)
        active_gameplay = latest_frame.state not in [GameState.NOT_PLAYED, GameState.GAME_OVER]

        prompt = (
            "You are controlling an ARC-AGI-3 environment.\n"
            "Return EXACTLY one token and nothing else.\n"
            "Allowed tokens: RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6.\n\n"
            "Hard rules:\n"
            "- If state is NOT_PLAYED or GAME_OVER, return RESET.\n"
            "- Otherwise NEVER return RESET.\n"
            "- During active gameplay, you MUST return one of ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6.\n"
            "- Do not explain your answer.\n\n"
            f"state={latest_frame.state.name}\n"
            f"active_gameplay={str(active_gameplay).lower()}\n"
            f"levels_completed={latest_frame.levels_completed}\n"
            f"available_actions={available_actions}\n"
            f"recent_actions={recent_actions}\n"
            f"{frame_summary}\n"
        )

        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[: self.max_prompt_chars] + "\n[TRUNCATED]"

        return prompt

    def _available_action_names(self, latest_frame: FrameData) -> list[str]:
        try:
            return [a.name for a in latest_frame.available_actions]
        except Exception:
            return []

    def _recent_actions(self, frames: list[FrameData]) -> list[str]:
        history: list[str] = []
        for frame in frames[-self.max_history :]:
            action_input = getattr(frame, "action_input", None)
            action_id = getattr(action_input, "id", None)
            if action_id is not None:
                history.append(action_id.name)
        return history

    def _summarize_frame(self, latest_frame: FrameData) -> str:
        frame = latest_frame.frame
        if not frame:
            return "frame_summary=empty"

        layers = len(frame)
        height = len(frame[0]) if frame and frame[0] else 0
        width = len(frame[0][0]) if height and frame[0][0] else 0

        colors = Counter()
        sample_rows: list[str] = []

        first_layer = frame[0] if frame else []
        for row_idx, row in enumerate(first_layer):
            colors.update(row)
            if row_idx < 6:
                sample_rows.append("".join(self._encode_cell(cell) for cell in row[:24]))

        top_colors = colors.most_common(6)
        return (
            f"frame_dims={layers}x{height}x{width}\n"
            f"top_colors={top_colors}\n"
            f"sample_rows={sample_rows}"
        )

    @staticmethod
    def _encode_cell(cell: Any) -> str:
        try:
            value = int(cell)
        except (TypeError, ValueError):
            return "?"
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if 0 <= value < len(alphabet):
            return alphabet[value]
        return "*"

    def _safe_parse_action(self, text: str, latest_frame: FrameData) -> GameAction:
        if not text:
            return self._fallback_action(latest_frame, "empty answer")

        upper = text.upper().strip()
        match = re.search(r"(RESET|ACTION[1-6])", upper)
        if not match:
            return self._fallback_action(latest_frame, f"invalid answer: {text[:80]}")

        name = match.group(1)
        action = GameAction.from_name(name)

        if latest_frame.state not in [GameState.NOT_PLAYED, GameState.GAME_OVER] and action is GameAction.RESET:
            return self._fallback_action(latest_frame, "guard blocked RESET during active gameplay")

        if action is GameAction.ACTION6:
            action.set_data({"x": 32, "y": 32})

        return action

    def _fallback_action(self, latest_frame: FrameData, reason: str) -> GameAction:
        self.last_fallback_reason = reason
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            action = GameAction.ACTION5

        if action.is_simple():
            action.reasoning = f"fallback: {reason}"
        logger.warning(
            "Aegean fallback for %s step=%s state=%s -> %s (%s)",
            self.game_id,
            self.step_count,
            latest_frame.state.name,
            action.name,
            reason,
        )
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
        if self._metrics_written:
            return
        self._metrics_written = True

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
            "last_action_name": self.last_action_name,
            "last_fallback_reason": self.last_fallback_reason[:160],
        }

        fieldnames = list(row.keys())
        with self._csv_lock:
            file_exists = csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    def _debug_log(
        self,
        *,
        step: int,
        state: str,
        raw_answer: str,
        action: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        error: str = "",
    ) -> None:
        if not self.debug:
            return
        logger.info(
            "Aegean step=%s game=%s state=%s raw=%r action=%s prompt_tokens=%s completion_tokens=%s latency=%.3fs error=%s",
            step,
            self.game_id,
            state,
            raw_answer[:120],
            action,
            prompt_tokens,
            completion_tokens,
            latency,
            error[:160],
        )
