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

    def start(self, now: float, blink_total: int = 0) -> None:
        self.window_start = now
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

        # Track yawning (estimated from mouth opening ratio)
        if mouth_open_ratio is not None and mouth_open_ratio > 50:  # Threshold for yawn detection
            self.yawning_episodes += 1

        # Track blinks
        if blink_total is not None:
            self.last_blink_total = int(blink_total)

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
