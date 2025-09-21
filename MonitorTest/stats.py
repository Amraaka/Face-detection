from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class StatsAggregator:
    interval_seconds: float = 5.0
    window_start: float = 0.0
    
    # Gaze tracking
    looking_forward_frames: int = 0
    looking_left_frames: int = 0
    looking_right_frames: int = 0
    looking_up_frames: int = 0
    looking_down_frames: int = 0
    total_frames: int = 0
    
    # Drowsiness tracking
    drowsiness_episodes: int = 0
    total_drowsy_duration: float = 0.0
    current_drowsy_start: Optional[float] = None
    
    # Phone usage tracking
    phone_usage_episodes: int = 0
    total_phone_duration: float = 0.0
    current_phone_start: Optional[float] = None
    
    # Yawning tracking (estimated from mouth opening)
    yawning_episodes: int = 0
    
    # Blink tracking
    blink_total_at_start: Optional[int] = None
    last_blink_total: int = 0
    
    # Time-series window state
    last_update_ts: float = 0.0
    # Eyes (episodes that reach 3s within window)
    eyes_closed_active: bool = False
    eyes_episode_accum_sec: float = 0.0
    eyes_episode_crossed: bool = False
    eyes_closed_over3_episodes_win: int = 0
    eyes_closed_over3_total_sec_win: float = 0.0
    
    # Head per-direction (over threshold 3s episodes)
    head_dirs: Dict[str, Dict[str, float | bool | int]] = None  # initialized in start()
    
    # Phone per-window accumulators
    phone_active_prev: bool = False
    phone_active_accum_sec: float = 0.0
    phone_started_in_window: int = 0
    phone_ended_in_window: int = 0

    def start(self, now: float, blink_total: int = 0) -> None:
        self.window_start = now
        self.last_update_ts = now
        self.looking_forward_frames = 0
        self.looking_left_frames = 0
        self.looking_right_frames = 0
        self.looking_up_frames = 0
        self.looking_down_frames = 0
        self.total_frames = 0
        self.drowsiness_episodes = 0
        self.total_drowsy_duration = 0.0
        self.current_drowsy_start = None
        self.phone_usage_episodes = 0
        self.total_phone_duration = 0.0
        self.current_phone_start = None
        self.yawning_episodes = 0
        self.blink_total_at_start = blink_total
        self.last_blink_total = blink_total
        # Time-series window trackers
        self.eyes_closed_active = False
        self.eyes_episode_accum_sec = 0.0
        self.eyes_episode_crossed = False
        self.eyes_closed_over3_episodes_win = 0
        self.eyes_closed_over3_total_sec_win = 0.0
        self.head_dirs = {
            "left": {"active": False, "accum": 0.0, "crossed": False, "episodes": 0, "total_sec": 0.0},
            "right": {"active": False, "accum": 0.0, "crossed": False, "episodes": 0, "total_sec": 0.0},
            "up": {"active": False, "accum": 0.0, "crossed": False, "episodes": 0, "total_sec": 0.0},
            "down": {"active": False, "accum": 0.0, "crossed": False, "episodes": 0, "total_sec": 0.0},
        }
        self.phone_active_prev = False
        self.phone_active_accum_sec = 0.0
        self.phone_started_in_window = 0
        self.phone_ended_in_window = 0

    def _categorize_gaze(self, yaw: float, pitch: float) -> str:
        """Categorize gaze direction based on yaw and pitch angles"""
        # Thresholds for gaze direction (in degrees)
        YAW_THRESHOLD = 15
        PITCH_THRESHOLD = 10
        
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

        # Delta time since last update for per-window accumulations
        dt = max(0.0, now - (self.last_update_ts or now))
        self.last_update_ts = now

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

        # Track drowsiness (eyes closed for extended periods)
        if eyes_closed:
            if self.current_drowsy_start is None:
                self.current_drowsy_start = now
        else:
            if self.current_drowsy_start is not None:
                drowsy_duration = now - self.current_drowsy_start
                if drowsy_duration >= 1.0:  # Consider drowsy if eyes closed for 1+ seconds
                    self.drowsiness_episodes += 1
                    self.total_drowsy_duration += drowsy_duration
                self.current_drowsy_start = None

        # Eyes closed state for time-series (>=3s episodes within window)
        if eyes_closed:
            # enter active if not
            if not self.eyes_closed_active:
                self.eyes_closed_active = True
                self.eyes_episode_accum_sec = 0.0
                self.eyes_episode_crossed = False
            # accumulate
            self.eyes_episode_accum_sec += dt
            if (not self.eyes_episode_crossed) and self.eyes_episode_accum_sec >= 3.0:
                self.eyes_closed_over3_episodes_win += 1
                self.eyes_episode_crossed = True
        else:
            # finalize current episode if any
            if self.eyes_closed_active:
                if self.eyes_episode_crossed:
                    self.eyes_closed_over3_total_sec_win += self.eyes_episode_accum_sec
                # reset
                self.eyes_closed_active = False
                self.eyes_episode_accum_sec = 0.0
                self.eyes_episode_crossed = False

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

        # Phone per-window accumulators
        if phone_detected and not self.phone_active_prev:
            # phone started now within this window
            self.phone_started_in_window += 1
        if (not phone_detected) and self.phone_active_prev:
            # phone ended now within this window
            self.phone_ended_in_window += 1
        self.phone_active_prev = phone_detected
        if phone_detected:
            self.phone_active_accum_sec += dt

        # Track yawning (estimated from mouth opening ratio)
        if mouth_open_ratio is not None and mouth_open_ratio > 50:  # Threshold for yawn detection
            self.yawning_episodes += 1

        # Track blinks
        if blink_total is not None:
            self.last_blink_total = int(blink_total)

        # Head over-threshold per-direction episode tracking (>=3s)
        # Determine which directions are currently over threshold
        over = {"left": False, "right": False, "up": False, "down": False}
        if yaw is not None and pitch is not None:
            # Using same thresholds as _categorize_gaze
            YAW_THRESHOLD = 15
            PITCH_THRESHOLD = 10
            if yaw > YAW_THRESHOLD:
                over["left"] = True
            if yaw < -YAW_THRESHOLD:
                over["right"] = True
            if pitch < -PITCH_THRESHOLD:
                over["up"] = True
            if pitch > PITCH_THRESHOLD:
                over["down"] = True

        for d in ["left", "right", "up", "down"]:
            state = self.head_dirs[d]
            if over[d]:
                if not state["active"]:
                    state["active"] = True
                    state["accum"] = 0.0
                    state["crossed"] = False
                state["accum"] += dt
                if (not state["crossed"]) and state["accum"] >= 3.0:
                    state["episodes"] = int(state["episodes"]) + 1
                    state["crossed"] = True
            else:
                if state["active"]:
                    if state["crossed"]:
                        state["total_sec"] = float(state["total_sec"]) + float(state["accum"]) 
                    state["active"] = False
                    state["accum"] = 0.0
                    state["crossed"] = False

    def ready(self, now: float) -> bool:
        return self.window_start != 0.0 and (now - self.window_start) >= self.interval_seconds

    def flush(self, now: float) -> dict:
        # Close any ongoing episodes
        if self.current_drowsy_start is not None:
            drowsy_duration = now - self.current_drowsy_start
            if drowsy_duration >= 1.0:
                self.drowsiness_episodes += 1
                self.total_drowsy_duration += drowsy_duration
        
        if self.current_phone_start is not None:
            phone_duration = now - self.current_phone_start
            self.phone_usage_episodes += 1
            self.total_phone_duration += phone_duration

        # Calculate percentages
        total_frames = max(self.total_frames, 1)
        gaze_percentages = {
            "forward": round((self.looking_forward_frames / total_frames) * 100, 1),
            "left": round((self.looking_left_frames / total_frames) * 100, 1),
            "right": round((self.looking_right_frames / total_frames) * 100, 1),
            "up": round((self.looking_up_frames / total_frames) * 100, 1),
            "down": round((self.looking_down_frames / total_frames) * 100, 1),
        }

        # Calculate blinks in window
        blinks_in_window = None
        if self.blink_total_at_start is not None:
            blinks_in_window = max(0, self.last_blink_total - self.blink_total_at_start)

        doc = {
            "timestamp": now,
            "window_start": self.window_start,
            "window_end": now,
            "duration_sec": round(now - self.window_start, 2),
            
            # Gaze analysis
            "gaze_direction_percentages": gaze_percentages,
            "attention_score": gaze_percentages["forward"],  # How much time looking forward
            
            # Drowsiness analysis
            "drowsiness_episodes": self.drowsiness_episodes,
            "total_drowsy_duration_sec": round(self.total_drowsy_duration, 2),
            "avg_drowsy_episode_duration": round(self.total_drowsy_duration / max(self.drowsiness_episodes, 1), 2),
            
            # Phone usage analysis
            "phone_usage_episodes": self.phone_usage_episodes,
            "total_phone_duration_sec": round(self.total_phone_duration, 2),
            "avg_phone_episode_duration": round(self.total_phone_duration / max(self.phone_usage_episodes, 1), 2),
            
            # Other behaviors
            "yawning_episodes": self.yawning_episodes,
            "blinks_count": blinks_in_window,
            "blink_rate_per_minute": round((blinks_in_window or 0) / (self.interval_seconds / 60), 1),
            
            # Overall assessment
            "distraction_level": self._calculate_distraction_level(gaze_percentages, self.phone_usage_episodes, self.drowsiness_episodes),
        }
        
        # Reset for next window
        self.start(now, self.last_blink_total)
        return doc

    def flush_timeseries(self, now: float) -> dict:
        """Produce a window document matching the required time-series schema.
        This method does NOT write to any store. Caller can pass the document to TimeSeriesStore.insert_window().
        """
        # Finalize any ongoing per-window episodes up to 'now'
        dt_tail = max(0.0, now - (self.last_update_ts or now))
        if dt_tail > 0:
            # Eyes
            if self.eyes_closed_active:
                self.eyes_episode_accum_sec += dt_tail
            # Phone
            if self.phone_active_prev:
                self.phone_active_accum_sec += dt_tail
            # Head
            for d in ["left", "right", "up", "down"]:
                if self.head_dirs[d]["active"]:
                    self.head_dirs[d]["accum"] += dt_tail

        # If crossing occurred by now, ensure counters updated
        if self.eyes_closed_active and (not self.eyes_episode_crossed) and self.eyes_episode_accum_sec >= 3.0:
            self.eyes_closed_over3_episodes_win += 1
            self.eyes_episode_crossed = True
        for d in ["left", "right", "up", "down"]:
            st = self.head_dirs[d]
            if st["active"] and (not st["crossed"]) and st["accum"] >= 3.0:
                st["episodes"] = int(st["episodes"]) + 1
                st["crossed"] = True

        # Complete totals for ongoing active segments
        if self.eyes_closed_active and self.eyes_episode_crossed:
            self.eyes_closed_over3_total_sec_win += self.eyes_episode_accum_sec
        for d in ["left", "right", "up", "down"]:
            st = self.head_dirs[d]
            if st["active"] and st["crossed"]:
                st["total_sec"] = float(st["total_sec"]) + float(st["accum"]) 

        window_start_s = self.window_start
        window_end_s = now
        duration = round(max(0.0, window_end_s - window_start_s), 3)

        # Calculate blinks in window
        blinks_in_window = 0
        if self.blink_total_at_start is not None:
            blinks_in_window = max(0, self.last_blink_total - self.blink_total_at_start)

        head_doc = {
            k: {"episodes": int(self.head_dirs[k]["episodes"]), "total_sec": round(float(self.head_dirs[k]["total_sec"]), 3)}
            for k in ["left", "right", "up", "down"]
        }

        doc = {
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
            "duration_sec": float(duration),

            # Eye
            "blinks_count": int(blinks_in_window),
            "eyes_closed_over_3s_episodes": int(self.eyes_closed_over3_episodes_win),
            "eyes_closed_over_3s_total_sec": round(float(self.eyes_closed_over3_total_sec_win), 3),

            # Head
            "head_over_3s": head_doc,

            # Phone
            "phone_episodes": int(self.phone_started_in_window + self.phone_ended_in_window),
            "phone_total_sec": round(float(self.phone_active_accum_sec), 3),
        }

        # Reset for next window
        self.start(now, self.last_blink_total)
        return doc

    def _calculate_distraction_level(self, gaze_percentages: Dict[str, float], phone_episodes: int, drowsy_episodes: int) -> str:
        """Calculate overall distraction level based on various factors"""
        attention_score = gaze_percentages["forward"]
        
        if phone_episodes > 0 or drowsy_episodes > 2:
            return "high"
        elif attention_score < 50:
            return "medium"
        elif attention_score >= 80:
            return "low"
        else:
            return "medium"
