# Aglow Recommendation Pipeline — Complex (Future-State)

> **Status:** This document describes the *target* recommender, not what's currently built. The shipped v1 lives in `recommendation_pipeline.md` and is a deliberately simpler WALS → filter → re-rank pipeline. Use this doc as the roadmap for future iterations.

## Data Landscape

| Signal | Table | Volume | Notes |
|--------|-------|--------|-------|
| Video watches | `feed.watched_at` | 80K watched / 88K shown | Best implicit signal — high volume |
| Watch count | `feed.watch_count` | per feed row | Repeat views = strong interest |
| Reactions | `feed.reaction` | 2,430 thumbs_up, 502 thumbs_down | Explicit signal — sparse but high value |
| Audio listens | `audio.listened_at` | 3,654 / 6,218 listened | 1:1 engagement signal |
| Contact graph | `contact` | 168K rows, 3,804 mutual (on-platform) | Social proximity |
| Video metadata | `video` | 6,744 videos, 97% have prompt_text | Content features (prompt, duration, location) |
| Prompts | `prompt` | Topic/question that videos respond to | Natural content grouping |
| Profiles | `profile` | 2,287 users, 614 active in feed | Location, utc_offset for time-aware recs |

**Key constraints:**
- 2,287 users total, ~600 active — classic cold-start / small-community problem
- Videos are short (~9 sec avg), prompt-driven
- Contact graph is mostly one-sided (phone contacts not on platform) — only 3,804 edges are usable
- Reactions are sparse (only 4.7% of feed items get a reaction)

---

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│  Data Sync  │────>│  Feature     │────>│  Scoring &    │────>│  Feed API  │
│  (Extract)  │     │  Store       │     │  Ranking      │     │  (Serve)   │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
     │                    │                     │                     │
  Supabase DB        User & Video          Candidate Gen +       Ranked list
  → read replica     embeddings &          Re-ranking             per user
                     interaction matrix
```

---

## Stage 1: Feature Engineering

### User Features
Computed per user from existing tables:

| Feature | Source | Logic |
|---------|--------|-------|
| `watch_rate` | feed | `watched / shown` per user |
| `like_rate` | feed | `thumbs_up / watched` per user |
| `preferred_prompts` | feed → video → prompt | Top prompt categories by watch rate |
| `active_hours` | profile.utc_offset + feed timestamps | When they engage |
| `social_degree` | contact | Count of on-platform contacts |
| `content_creator` | video | Boolean — has uploaded videos? |
| `avg_watch_depth` | feed.watch_count | Repeat view tendency |

### Video Features

| Feature | Source | Logic |
|---------|--------|-------|
| `prompt_category` | video.prompt_text / prompt.text | Group by prompt topic |
| `global_watch_rate` | feed | `watched / shown` across all users |
| `like_ratio` | feed | `thumbs_up / (thumbs_up + thumbs_down)` |
| `duration_bucket` | video.duration | short (<5s), medium (5-15s), long (15s+) |
| `creator_popularity` | feed + video | Avg watch rate of creator's videos |
| `recency` | video.created_at | Decay factor for freshness |
| `location_cluster` | video.location | Geographic grouping |

### Interaction Matrix
A sparse matrix of `(user, video) → score`:

```
score = watched * 1.0
      + (watch_count > 1) * 0.5
      + thumbs_up * 2.0
      + thumbs_down * -3.0
      + sent_audio_to_creator * 1.5
```

### User cohort labels (via PCA on user features)

PCA on the user-feature matrix yields interpretable axes:
- **PC1** — overall activity
- **PC2** — watch_rate ↑ vs contacts ↓ ("engaged loner" axis)
- **PC3** — reaction intensity
- **PC4** — on-platform contact ratio

Cohort labels (e.g. k-means in PC space) feed forward to personalize the candidate strategy mix and re-rank coefficients.

---

## Stage 2: Candidate Generation

Given the small user base, a multi-strategy approach avoids filter bubbles:

### Strategy A — Collaborative Filtering (40% of candidates by default)
- WALS (Weighted Alternating Least Squares) via `implicit.als.AlternatingLeastSquares` on the interaction matrix
- At 600 active users and 6K videos, this fits in memory easily
- Produces: "users who watched similar videos also watched X"
- Outputs both user embeddings $U \in \mathbb{R}^{n_u \times k}$ and video embeddings $V \in \mathbb{R}^{n_v \times k}$, with predicted affinity $\hat{R}_{uv} = u_u^\top v_v$

### Strategy B — Social Graph (30% of candidates by default)
- Surface videos from a user's on-platform contacts
- Weight by `contact.last_interaction_at` recency
- Produces: "your friends watched/created X"

### Strategy C — Content-Based (20% of candidates by default)
- Match user's `preferred_prompts` to unwatched videos with the same prompt
- Use text similarity on `prompt_text` / `uploaded_text` (MiniLM embedding) for finer matching
- Produces: "you liked videos about topic X, here's more"

### Strategy D — Explore / Cold Start (10% of candidates by default)
- Popular videos the user hasn't seen (global watch_rate + like_ratio)
- New videos (< 48 hrs) with minimum quality `score`
- Ensures new users and new content get exposure

**Per-user mix:** the 40/30/20/10 split is the cold-start default. Real mix is driven by PCA cohort — e.g. a user with no contacts gets B's 30% reallocated to D; a high-activity user gets D shrunk and A expanded.

---

## Stage 3: Filtering (hard eligibility)

Run before scoring/re-ranking to avoid wasting compute on items that will be removed anyway. Deterministic yes/no checks:

- `video.is_deleted = true` → drop
- `video.is_public = false` → drop (unless from a contact)
- creator in user's `contact.is_blocked` → drop
- video already in user's watch history → drop
- `feed.reaction = 'reported'` count above threshold → drop

---

## Stage 4: Re-ranking

Take the filtered candidate set and re-rank:

```python
final_score = (
    candidate_score * 0.4          # from generation strategy
    + recency_decay * 0.2          # prefer fresh content
    + social_boost * 0.2           # boost if from contact
    + diversity_penalty * 0.1      # penalize same prompt/creator in a row
    + creator_variety * 0.1        # avoid over-indexing on one creator
)
```

Coefficients themselves are personalized off the PCA cohort.

**Soft rules applied after sort:**
- Cap any single creator to max 2 videos in a 10-video feed page
- Boost videos matching the user's timezone active hours
- Prompt-diversity penalty for consecutive same-prompt videos

---

## Stage 5: Serving

| Option | Complexity | Fit |
|--------|-----------|-----|
| **Precompute + cache** | Low | Best for now — batch job writes top 50 recs per user to a `recommendation` table, refreshed every 1-6 hours |
| Real-time scoring | Medium | Overkill at current scale |
| Hybrid | High | Future state |

Serving flow:
1. Cron job runs pipeline (extract → score → write)
2. Feed API reads from `recommendation` table, ordered by `final_score`
3. Client paginates through the list
4. Impressions and reactions feed back into the next pipeline run

---

## Implementation Plan

### Phase 1 — Baseline (week 1-2) ✅ partially done in simple v1
- Build the feature queries as SQL views or a Python ETL script
- Implement Strategy D only (popularity-based) as the initial ranker
- Write results to a `recommendation` table in Supabase
- Measure: watch_rate of recommended vs current feed

### Phase 2 — Collaborative + Social (week 3-4) — simple v1 covers Strategy A
- Build the interaction matrix from feed data ✅ in simple v1
- Add WALS collaborative filtering (use `implicit` library) ✅ in simple v1
- Add social graph boosting from contact table — *deferred*
- A/B test against Phase 1

### Phase 3 — Content Matching (week 5-6)
- Embed `prompt_text` / `uploaded_text` with a lightweight model (e.g. `all-MiniLM-L6-v2`)
- Add content-based candidate generation
- Tune the re-ranking weights based on A/B results

### Phase 4 — Cohort Personalization
- PCA on user features → cohort labels
- Personalize strategy mix and re-rank coefficients per cohort
- Hold out users and measure NDCG@10, recall@50 per cohort

### Phase 5 — Feedback Loop (ongoing)
- Log impressions (shown but not watched) to measure recall
- Track recommendation → watch → reaction funnel
- Retrain/retune weekly

---

## Metrics to Track

| Metric | Definition | Target |
|--------|-----------|--------|
| **Watch rate** | `watched / shown` for recommended videos | > 30% (current ~91% overall but likely lower for non-social content) |
| **Reaction rate** | `reacted / watched` | > 8% |
| **Negative reaction rate** | `thumbs_down / reacted` | < 15% |
| **Creator coverage** | Unique creators shown / total creators | > 60% |
| **Prompt diversity** | Unique prompts in a user's feed per day | > 3 |
| **Audio follow-ups** | Audio messages sent after watching a rec | Track trend |
