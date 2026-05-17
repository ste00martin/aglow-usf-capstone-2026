# Aglow Recommendation Pipeline — Simple v1

> **Status:** This is the shipped v1. The richer 4-strategy future-state design with PCA-cohort personalization lives in `recommendation_pipeline_complex.md`.

A deliberately small pipeline: one candidate-generation method (WALS), hard filtering, and a fixed-coefficient re-rank. Three linear stages, no per-user personalization beyond what WALS already learns.

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   1. WALS    │──▶│  2. Filter   │──▶│ 3. Re-rank   │──▶ recommendations.csv
│ candidate    │   │ hard         │   │ fixed coefs  │
│ generation   │   │ eligibility  │   │ + creator cap│
└──────────────┘   └──────────────┘   └──────────────┘
```

Implemented in `recommend.py`.

---

## Stage 1 — WALS candidate generation

Build a sparse user × video interaction matrix from `feed` (+ `audio`):

```
R[u, v] = watched * 1.0
       + (watch_count > 1) * 0.5
       + thumbs_up * 2.0
       + thumbs_down * -3.0
       + sent_audio_to_creator * 1.5
```

Negative values clipped to 0 before WALS (implicit feedback assumes non-negative confidence — thumbs_down handling moves to the filter stage where reported items get dropped explicitly).

Fit WALS with `implicit.als.AlternatingLeastSquares`:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `factors` | 32 | Small dataset, 32 latent dims avoids overfitting |
| `regularization` | 0.05 | Default; tune later |
| `alpha` | 40 | Confidence scale from Hu/Koren paper |
| `iterations` | 20 | Loss flattens by ~15 iters at this scale |

For each user, pull top-200 candidates by `model.recommend(u, R[u], N=200, filter_already_liked_items=False)`. We keep already-watched items in the candidate pool and let the filter stage drop them — keeps stages cleanly separated.

**Cold-start:** users with zero interactions are skipped in v1. (Future: lookalike via profile features → covered in the complex doc.)

---

## Stage 2 — Filter (hard eligibility)

Deterministic drop list. Pre-load once, apply per-user:

| Rule | Source |
|---|---|
| `video.is_deleted = true` | `video` |
| `video.is_public = false` | `video` |
| creator in user's blocked contacts (`contact.is_blocked = true`) | `contact` |
| video already in user's watched set (`feed.watched_at IS NOT NULL`) | `feed` |
| user has marked video with `reaction = 'reported'` | `feed` |

Filtering runs **before** re-ranking — no compute wasted scoring videos that will be removed.

---

## Stage 3 — Re-rank (fixed coefficients)

For each surviving candidate:

```
final_score = 0.6 * normalized_cand_score
            + 0.4 * recency_decay
```

where:
- `normalized_cand_score` = WALS score / max WALS score across that user's candidate pool, so it sits in [0, 1]
- `recency_decay` = `exp(-age_days / 14)` using `video.created_at`

Sort descending, then apply **one soft rule**:
- **Creator cap:** walk the sorted list, drop any video whose creator already has 2 entries above it.

Truncate to top 50 per user.

---

## Output

`recommendations.csv` — columns: `user_id, video_id, rank, final_score`. Approximate shape: ~600 active users × 50 recs ≈ 30K rows.

Not yet written back to Supabase (`recommendation` table) — that's a separate deployment concern.

---

## Out of scope for v1

Everything below is deferred to the complex version (`recommendation_pipeline_complex.md`):
- Social-graph candidate strategy
- Content-based / prompt-text-embedding strategy
- Popularity / fresh-content explore strategy
- PCA-cohort-personalized strategy mix
- PCA-cohort-personalized re-rank coefficients
- Soft rules beyond creator cap (prompt diversity, timezone boost, social boost)
- Cold-start handling for users with no interaction history
- Held-out evaluation / NDCG metrics
- Writing back to a Supabase `recommendation` table + cron schedule

---

## Running it

```bash
./venv/bin/pip install implicit          # one-time
./venv/bin/python recommend.py            # end-to-end, writes recommendations.csv
```

Pipeline prints per-stage timing and row counts. End-to-end runtime at current scale: seconds, not minutes.
