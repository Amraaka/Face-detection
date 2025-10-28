from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timezone


@dataclass
class StatsAggregator:
    interval_seconds: float = 5.0
    session_id: Optional[str] = None
    window_start: float = 0.0
    
    looking_forward_frames: int = 0
    looking_left_frames: int = 0
    looking_right_frames: int = 0
    looking_up_frames: int = 0
    looking_down_frames: int = 0
    total_frames: int = 0
    
    eyes_closed_3s_episodes: int = 0
    eyes_closed_3s_total: float = 0.0
    current_eyes_closed_start: Optional[float] = None
    
    phone_usage_episodes: int = 0
    total_phone_duration: float = 0.0
    current_phone_start: Optional[float] = None
    
    head_3s_episodes: Dict[str, int] = None
    head_3s_totals: Dict[str, float] = None
    current_head_dir: Optional[str] = None
    current_head_start: Optional[float] = None
    
    blink_total_at_start: Optional[int] = None
    last_blink_total: int = 0
    
    # Emotion tallies within the current window
    emotion_counts: Optional[Dict[str, int]] = None
    
    # Activity tallies within the current window
    activity_counts: Optional[Dict[str, int]] = None
    activity_last_label: Optional[str] = None
    activity_confidences: Optional[List[float]] = None

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
        # reset emotion counts each window
        self.emotion_counts = {"Angry": 0, "Happy": 0, "Neutral": 0}
        # reset activity counts each window
        self.activity_counts = {"drinking": 0, "other_activities": 0, "talking_phone": 0, "yawning": 0}
        self.activity_last_label = None
        self.activity_confidences = []

    def _categorize_gaze(self, yaw: float, pitch: float) -> str:
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
            return "forward"  

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
        emotion_label: Optional[str] = None,
        activity_label: Optional[str] = None,
        activity_confidence: Optional[float] = None,
    ) -> None:
        if self.window_start == 0.0:
            self.start(now, blink_total or 0)

        self.total_frames += 1

        # Tally emotions if provided and recognized
        if emotion_label and self.emotion_counts is not None:
            if emotion_label in self.emotion_counts:
                self.emotion_counts[emotion_label] += 1
        
        # Tally activities if provided and recognized
        if activity_label and self.activity_counts is not None:
            if activity_label in self.activity_counts:
                self.activity_counts[activity_label] += 1
            self.activity_last_label = activity_label
        
        # Store activity confidence for averaging
        if activity_confidence is not None and self.activity_confidences is not None:
            self.activity_confidences.append(activity_confidence)

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

        if eyes_closed:
            if self.current_eyes_closed_start is None:
                self.current_eyes_closed_start = now
        else:
            if self.current_eyes_closed_start is not None:
                closed_dur = now - self.current_eyes_closed_start
                if closed_dur >= 1.0:
                    self.eyes_closed_3s_episodes += 1
                    self.eyes_closed_3s_total += closed_dur
                self.current_eyes_closed_start = None

        if phone_detected:
            if self.current_phone_start is None:
                self.current_phone_start = now
        else:
            if self.current_phone_start is not None:
                phone_duration = now - self.current_phone_start
                self.phone_usage_episodes += 1
                self.total_phone_duration += phone_duration
                self.current_phone_start = None

        direction = None
        if yaw is not None and pitch is not None:
            cat = self._categorize_gaze(yaw, pitch)
            if cat in ("left", "right", "up", "down"):
                direction = cat
        if direction:
            if self.current_head_dir is None:
                self.current_head_dir = direction
                self.current_head_start = now
            elif self.current_head_dir != direction:
                dur = now - (self.current_head_start or now)
                if dur >= 1.0:
                    self.head_3s_episodes[self.current_head_dir] += 1
                    self.head_3s_totals[self.current_head_dir] += dur
                self.current_head_dir = direction
                self.current_head_start = now
        else:
            if self.current_head_dir is not None and self.current_head_start is not None:
                dur = now - self.current_head_start
                if dur >= 1.0:
                    self.head_3s_episodes[self.current_head_dir] += 1
                    self.head_3s_totals[self.current_head_dir] += dur
                self.current_head_dir = None
                self.current_head_start = None

        if blink_total is not None:
            self.last_blink_total = int(blink_total)

    def ready(self, now: float) -> bool:
        return self.window_start != 0.0 and (now - self.window_start) >= self.interval_seconds

    def flush(self, now: float) -> dict:
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

        blinks_in_window = None
        if self.blink_total_at_start is not None:
            blinks_in_window = max(0, self.last_blink_total - self.blink_total_at_start)

        # Calculate average activity confidence and find top activity
        activity_top_label = None
        activity_top_conf_avg = None
        if self.activity_counts and any(count > 0 for count in self.activity_counts.values()):
            # Find the most frequent activity
            max_count = max(self.activity_counts.values())
            for label, count in self.activity_counts.items():
                if count == max_count:
                    activity_top_label = label
                    break
            
            # Calculate average confidence
            if self.activity_confidences:
                activity_top_conf_avg = round(sum(self.activity_confidences) / len(self.activity_confidences), 4)

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
            "emotion_counts": dict(self.emotion_counts or {}),
            
            # Activity data
            "activity_counts": dict(self.activity_counts or {}),
            "activity_top_label": activity_top_label,
            "activity_top_conf_avg": activity_top_conf_avg,
            "activity_last": self.activity_last_label,
        }
        
        self.start(now, self.last_blink_total)
        return doc

