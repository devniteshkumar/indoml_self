# Submission Specification

**Format**: `.jsonl` (One JSON object per line)[cite: 1].
**Required Fields**:
- `clip_id` (string): Must exactly match the withheld test clip IDs.
- `events` (list of dictionaries): Contains detected events.
  - `onset` (float): Start time in seconds[cite: 1].
  - `offset` (float): End time in seconds[cite: 1].

**Notes**:
- If no events are detected, pass an empty list `[]`.
- Do NOT fabricate submission schemas[cite: 1].