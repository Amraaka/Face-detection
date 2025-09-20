import pygame

class AlertPlayer:
    def __init__(self, warning_sound_path="warning.mp3", debounce_seconds=3.0):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound(warning_sound_path)
        self.debounce_seconds = debounce_seconds
        self._last_time = 0.0

    def play(self, now: float):
        if now - self._last_time > self.debounce_seconds:
            self.sound.play()
            self._last_time = now