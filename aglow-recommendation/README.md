# aglow-recommendation

Recommender pipeline for the aglow feed. Two design tiers documented; the simple tier is implemented and runnable end-to-end.

## Contents

| File | Purpose |
|---|---|
| `recommendation_pipeline.md` | **Simple v1 spec** — shipped design: WALS → filter → re-rank with fixed coefficients |
| `recommendation_pipeline_complex.md` | **Future-state design** — full 4-strategy pipeline with PCA-cohort personalization |
| `recommend.py` | Simple v1 implementation — loads interaction matrix, fits WALS, filters, re-ranks, writes `recommendations.csv` |
| `pca_users.py` | Diagnostic — PCA on user-level engagement features (used for cohort labeling in the complex design) |
| `explore_tables.ipynb` | Schema exploration notebook — sample rows + row counts for each table |
| `requirements.txt` | Python dependencies |

## Pipeline shape (simple v1)

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   1. WALS    │──▶│  2. Filter   │──▶│ 3. Re-rank   │──▶ recommendations.csv
│ candidate    │   │ hard         │   │ fixed coefs  │
│ generation   │   │ eligibility  │   │ + creator cap│
└──────────────┘   └──────────────┘   └──────────────┘
```

See `recommendation_pipeline.md` for the spec and `recommend.py` for the implementation.

## Running

Requires a `.env` file (not committed) with a Supabase Postgres connection string:

```
DATABASE_URL=postgresql://user:pass@host:port/dbname
```

Then:

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python recommend.py
```

Output: `recommendations.csv` (~30K rows: user_id, video_id, rank, final_score).

End-to-end runtime at current data scale (~600 active users, ~6K videos, ~80K interactions): **under 4 seconds**.

## Verified behavior

The shipped v1 passes these correctness checks on real data:

- All 50 recommended video_ids are distinct per user
- `final_score` is monotone non-increasing within a user's ranked list
- No already-watched videos appear in any user's recommendations
- No `is_deleted` or `is_public = false` videos leak through
- Creator cap holds: no creator appears more than 2× in any user's top-50

## Status & next steps

- **Done:** Stage 1 (WALS) + Stage 2 (filter) + Stage 3 (re-rank), all running end-to-end on live Supabase data.
- **Next (in order):** offline evaluation (temporal split + recall@K / NDCG@K vs popularity baseline), then the complex-version features in `recommendation_pipeline_complex.md`.

## What's intentionally not here

- `.env` — credentials, never committed
- `recommendations.csv` — pipeline output containing real user/video UUIDs; treat as a build artifact, regenerate locally
- `venv/` — virtualenv
