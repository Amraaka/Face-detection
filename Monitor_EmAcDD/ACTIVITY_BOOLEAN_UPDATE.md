# Activity Detection Storage Update

## Problem
Previously, the system was counting **every frame** where an activity was detected, which could result in hundreds of counts in just a 5-second window. For example, if someone was yawning for 2 seconds at 30fps, it would count 60 instances of yawning in one 5-second window.

## Solution
Changed from **counting occurrences** to **boolean flags** (detected or not detected) within each time window.

## What Changed

### 1. Monitor_EmAc/stats.py

**Before:**
```python
activity_counts: Optional[Dict[str, int]] = None  # Counted every frame
```

**After:**
```python
activity_detected: Optional[Dict[str, bool]] = None  # Boolean flags
activity_last_detected_time: Optional[Dict[str, float]] = None  # Debouncing
```

**New Behavior:**
- Each activity has a boolean flag (True/False) instead of a count
- An activity is marked as `True` if detected at least once in the window
- **Debouncing:** After marking an activity as detected, it won't be marked again for 1 second (prevents rapid re-detection)
- This reduces storage from potentially 100+ counts to just 1 boolean per activity

### 2. Data Structure

**Old Format (stored in MongoDB):**
```json
{
  "activity_counts": {
    "drinking": 0,
    "other_activities": 6,
    "talking_phone": 11,
    "yawning": 123
  }
}
```

**New Format:**
```json
{
  "activity_detected": {
    "drinking": false,
    "other_activities": true,
    "talking_phone": true,
    "yawning": true
  }
}
```

### 3. Backend API Update

The `/api/stats/activities` endpoint now:
- Reads `activity_detected` (boolean flags) instead of `activity_counts`
- **Counts how many 5-second windows** had each activity detected
- Returns: number of windows where activity was `true`

**Example:**
- 6 windows in a 1-minute interval
- Yawning detected in 3 of those windows → returns `"yawning": 3`
- This represents "yawning happened in 3 different 5-second periods"

### 4. Database Schema

Updated `backend/src/models/DriverStat.js`:
```javascript
// Legacy support (backward compatible)
activity_counts: { type: Map, of: Number, default: undefined },

// New field (boolean flags)
activity_detected: { type: Map, of: Boolean, default: undefined },
```

## Benefits

1. **Reduced Data Storage:** Boolean flags take much less space than large counts
2. **More Meaningful Metrics:** "Activity happened" vs "Activity detected 123 times in 5 seconds"
3. **Better Time Resolution:** Shows activity occurrence across time bins, not just raw counts
4. **Debouncing:** 1-second minimum between detections prevents noise
5. **Backward Compatible:** Old data with `activity_counts` still works

## Usage

### Running the Monitor
```bash
python3 -m Monitor_EmAc.main --cam 1
```

### API Response Example
```bash
curl "http://localhost:4000/api/stats/activities?lastHours=24&interval=1m"
```

Response shows count of **windows** where activity was detected:
```json
{
  "timeseries": [
    {
      "ts": "2025-10-28T12:02:00.000Z",
      "activities": {
        "drinking": 0,      // 0 windows with drinking
        "other_activities": 1,  // 1 window with other activities
        "talking_phone": 1,     // 1 window with phone talking
        "yawning": 1            // 1 window with yawning
      }
    }
  ]
}
```

## Testing

1. Start the monitor: `python3 -m Monitor_EmAc.main --cam 1`
2. Perform activities (yawn, drink water, talk on phone)
3. Check the API: `curl "http://localhost:4000/api/stats/activities?lastHours=1&interval=1m"`
4. Verify numbers are reasonable (0-N windows) not hundreds

## Notes

- Each 5-second window can only have `true` or `false` for each activity
- The API aggregates these boolean flags across time bins
- Frontend charts now show "number of time periods with activity" instead of raw frame counts
