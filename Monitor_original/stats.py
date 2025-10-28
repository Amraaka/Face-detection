from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timezone


@dataclass
class StatsAggregator:
    interval_seconds: float = 5.0
    session_id: Optional[str] = None
    window_start: float = 0.0
    
    # Gaze tracking
    looking_forward_frames: int = 0
    looking_left_frames: int = 0
    looking_right_frames: int = 0
    looking_up_frames: int = 0
    looking_down_frames: int = 0
    total_frames: int = 0
    
    # Eyes closed episodes (>= 3s)
    eyes_closed_3s_episodes: int = 0
    eyes_closed_3s_total: float = 0.0
    current_eyes_closed_start: Optional[float] = None
    
    # Phone usage tracking
    phone_usage_episodes: int = 0
    total_phone_duration: float = 0.0
    current_phone_start: Optional[float] = None
    
    # Head turned (>= threshold) episodes per direction (>= 3s)
    head_3s_episodes: Dict[str, int] = None
    head_3s_totals: Dict[str, float] = None
    current_head_dir: Optional[str] = None
    current_head_start: Optional[float] = None
    
    # Blink tracking
    blink_total_at_start: Optional[int] = None
    last_blink_total: int = 0

    def start(self, now: float, blink_total: int = 0) -> None:
        self.window_start = now
        self.looking_forward_frames = 0
        self.looking_left_frames = 0
        self.looking_right_frames = 0
        self.looking_up_frames = 0
        self.looking_down_frames = 0
        self.total_frames = 0
        self.eyes_closed_3s_episodes = 0
        self.eyes_closed_3s_total = 0.0
        self.current_eyes_closed_start = None
        self.phone_usage_episodes = 0
        self.total_phone_duration = 0.0
        self.current_phone_start = None
        self.head_3s_episodes = {"left": 0, "right": 0, "up": 0, "down": 0}
        self.head_3s_totals = {"left": 0.0, "right": 0.0, "up": 0.0, "down": 0.0}
        self.current_head_dir = None
        self.current_head_start = None
        self.blink_total_at_start = blink_total
        self.last_blink_total = blink_total

    def _categorize_gaze(self, yaw: float, pitch: float) -> str:
        """Categorize gaze direction based on yaw and pitch angles"""
        # Align thresholds with HeadPoseEstimator defaults
        YAW_THRESHOLD = 20
        PITCH_THRESHOLD = 20
        
        if abs(yaw) <= YAW_THRESHOLD and abs(pitch) <= PITCH_THRESHOLD:
            return "forward"
        elif yaw > YAW_THRESHOLD:
            return "left"
        elif yaw < -YAW_THRESHOLD:
            return "right"
        elif pitch > PITCH_THRESHOLD:
            return "down"
        elif pitch < -PITCH_THRESHOLD:
            return "up"
        else:
            return "forward"  # default

    def update(
        self,
        now: float,
        *,
        yaw: Optional[float],
        pitch: Optional[float],
        phone_detected: bool,
        blink_total: Optional[int],
        eyes_closed: bool = False,
        mouth_open_ratio: Optional[float] = None,
    ) -> None:
        if self.window_start == 0.0:
            self.start(now, blink_total or 0)

        self.total_frames += 1

        # Track gaze direction
        if yaw is not None and pitch is not None:
            gaze_direction = self._categorize_gaze(yaw, pitch)
            if gaze_direction == "forward":
                self.looking_forward_frames += 1
            elif gaze_direction == "left":
                self.looking_left_frames += 1
            elif gaze_direction == "right":
                self.looking_right_frames += 1
            elif gaze_direction == "up":
                self.looking_up_frames += 1
            elif gaze_direction == "down":
                self.looking_down_frames += 1

        # Track eyes-closed episodes (>= 3s)
        if eyes_closed:
            if self.current_eyes_closed_start is None:
                self.current_eyes_closed_start = now
        else:
            if self.current_eyes_closed_start is not None:
                closed_dur = now - self.current_eyes_closed_start
                if closed_dur >= 3.0:
                    self.eyes_closed_3s_episodes += 1
                    self.eyes_closed_3s_total += closed_dur
                self.current_eyes_closed_start = None

        # Track phone usage episodes
        if phone_detected:
            if self.current_phone_start is None:
                self.current_phone_start = now
        else:
            if self.current_phone_start is not None:
                phone_duration = now - self.current_phone_start
                self.phone_usage_episodes += 1
                self.total_phone_duration += phone_duration
                self.current_phone_start = None

        # Track head-turned episodes per direction (>= 3s)
        direction = None
        if yaw is not None and pitch is not None:
            # Use same categorization/thresholds
            cat = self._categorize_gaze(yaw, pitch)
            if cat in ("left", "right", "up", "down"):
                direction = cat
        if direction:
            if self.current_head_dir is None:
                self.current_head_dir = direction
                self.current_head_start = now
            elif self.current_head_dir != direction:
                # Close previous episode and start new
                dur = now - (self.current_head_start or now)
                if dur >= 3.0:
                    self.head_3s_episodes[self.current_head_dir] += 1
                    self.head_3s_totals[self.current_head_dir] += dur
                self.current_head_dir = direction
                self.current_head_start = now
        else:
            # Head back to forward; close any ongoing
            if self.current_head_dir is not None and self.current_head_start is not None:
                dur = now - self.current_head_start
                if dur >= 3.0:
                    self.head_3s_episodes[self.current_head_dir] += 1
                    self.head_3s_totals[self.current_head_dir] += dur
                self.current_head_dir = None
                self.current_head_start = None

        # Track blinks
        if blink_total is not None:
            self.last_blink_total = int(blink_total)

    def ready(self, now: float) -> bool:
        return self.window_start != 0.0 and (now - self.window_start) >= self.interval_seconds

    def flush(self, now: float) -> dict:
        # Close any ongoing episodes
        if self.current_eyes_closed_start is not None:
            closed_dur = now - self.current_eyes_closed_start
            if closed_dur >= 3.0:
                self.eyes_closed_3s_episodes += 1
                self.eyes_closed_3s_total += closed_dur
        
        if self.current_phone_start is not None:
            phone_duration = now - self.current_phone_start
            self.phone_usage_episodes += 1
            self.total_phone_duration += phone_duration

        if self.current_head_dir is not None and self.current_head_start is not None:
            dur = now - self.current_head_start
            if dur >= 3.0:
                self.head_3s_episodes[self.current_head_dir] += 1
                self.head_3s_totals[self.current_head_dir] += dur

        # Calculate blinks in window
        blinks_in_window = None
        if self.blink_total_at_start is not None:
            blinks_in_window = max(0, self.last_blink_total - self.blink_total_at_start)

        # Build minimal time-series-friendly document per requirements
        window_start_s = float(self.window_start)
        window_end_s = float(now)
        doc = {
            "session_id": self.session_id,
            "window_start_dt": datetime.fromtimestamp(window_start_s, tz=timezone.utc),
            "window_end_dt": datetime.fromtimestamp(window_end_s, tz=timezone.utc),
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "duration_sec": round(window_end_s - window_start_s, 2),

            "blinks_count": blinks_in_window if blinks_in_window is not None else 0,
            "eyes_closed_over_3s_episodes": self.eyes_closed_3s_episodes,
            "eyes_closed_over_3s_total_sec": round(self.eyes_closed_3s_total, 2),

            "head_over_3s": {
                "left":  {"episodes": self.head_3s_episodes.get("left", 0),  "total_sec": round(self.head_3s_totals.get("left", 0.0), 2)},
                "right": {"episodes": self.head_3s_episodes.get("right", 0), "total_sec": round(self.head_3s_totals.get("right", 0.0), 2)},
                "up":    {"episodes": self.head_3s_episodes.get("up", 0),    "total_sec": round(self.head_3s_totals.get("up", 0.0), 2)},
                "down":  {"episodes": self.head_3s_episodes.get("down", 0),  "total_sec": round(self.head_3s_totals.get("down", 0.0), 2)},
            },

            "phone_episodes": self.phone_usage_episodes,
            "phone_total_sec": round(self.total_phone_duration, 2),
        }
        
        # Reset for next window
        self.start(now, self.last_blink_total)
        return doc

