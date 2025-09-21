# Time-series storage schema

Here’s a minimal, time-series–friendly structure matching only your current features. Store all time fields as UTC Date for range queries, and duplicate seconds as floats for math.

## Collections

1. sessions (one per run)
- session_id (UUID string)
- session_tz (IANA tz, e.g., Asia/Seoul)
- started_at_dt (UTC Date), ended_at_dt (UTC Date)

2. windows_5s (one doc per 5-second window)
- identity/time
  - session_id (string)
  - window_start_dt (UTC Date), window_end_dt (UTC Date)
  - window_start_s (float), window_end_s (float)
  - duration_sec (float)
- eye
  - blinks_count (int) // blinks detected within this 5s window
  - eyes_closed_over_3s_episodes (int) // episodes that crossed 3s within this window
  - eyes_closed_over_3s_total_sec (float) // summed seconds from those episodes within this window
- head (per-direction over-threshold)
  - head_over_3s: { left: { episodes: int, total_sec: float }, right: { episodes: int, total_sec: float }, up: { episodes: int, total_sec: float }, down: { episodes: int, total_sec: float } }
- phone
  - phone_episodes (int) // count of phone detections that started/ended in this window
  - phone_total_sec (float) // seconds with phone detected within this window

## Example document

```
{
  "session_id": "7d9c7b5a-5f61-4c2e-bf4a-0f3a7f28a9c2",
  "window_start_dt": "2025-09-22T10:00:00Z",
  "window_end_dt":   "2025-09-22T10:00:05Z",
  "window_start_s": 1758535200.0,
  "window_end_s":   1758535205.0,
  "duration_sec": 5.0,

  "blinks_count": 1,
  "eyes_closed_over_3s_episodes": 0,
  "eyes_closed_over_3s_total_sec": 0.0,

  "head_over_3s": {
    "left":  { "episodes": 0, "total_sec": 0.0 },
    "right": { "episodes": 1, "total_sec": 3.4 },
    "up":    { "episodes": 0, "total_sec": 0.0 },
    "down":  { "episodes": 0, "total_sec": 0.0 }
  },

  "phone_episodes": 1,
  "phone_total_sec": 1.6
}
```

## Indexes

- windows_5s: compound (session_id, window_start_dt), plus single-field on window_end_dt desc.
- sessions: index on started_at_dt.

## How to enable in this project

- Use the `MonitorTest` app with the `--timeseries` flag and supply your MongoDB Atlas URI (or env var `MONGODB_URI`). The code writes to two collections: `sessions` and `windows_5s`.
- Session documents are created at startup and marked ended on exit.
- Each 5s window flushed from the `StatsAggregator` is inserted into `windows_5s` with both UTC Date and float-second duplicates of window bounds.

Example run:

```
python -m MonitorTest.main --cam 0 --timeseries --mongo-uri "$MONGODB_URI"
```

Environment overrides (optional):
- `MONGODB_DB` (default: `driver_monitor`)
- `MONGODB_SESSIONS_COLL` (default: `sessions`)
- `MONGODB_WINDOWS_COLL` (default: `windows_5s`)
- `SESSION_TZ` (default: `UTC`)
- `STATS_INTERVAL` (default: `5` seconds)

Notes:
- All Date fields are stored as UTC. Float seconds are epoch seconds for quick numeric math/aggregations.
- Indexes are ensured automatically on startup by `TimeSeriesStore`.
