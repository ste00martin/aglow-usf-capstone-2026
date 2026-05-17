"""Simple v1 aglow recommender: WALS → Filter → Re-rank.

Pipeline:
  1. Build sparse user×video interaction matrix from feed + audio.
  2. Fit WALS (implicit.als.AlternatingLeastSquares).
  3. Generate top-200 candidates per user.
  4. Apply hard eligibility filters (deleted, private, blocked, watched, reported).
  5. Re-rank with fixed coefficients and cap each creator to 2 per user.
  6. Write top-50 per user to recommendations.csv.

See recommendation_pipeline.md for the spec.
"""
import os
import time
import warnings
from collections import defaultdict

import implicit
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine, exc as sa_exc, text

warnings.filterwarnings("ignore", category=sa_exc.SAWarning)

# Re-rank coefficients (see recommendation_pipeline.md Stage 3)
W_CAND = 0.6
W_RECENCY = 0.4
RECENCY_HALF_LIFE_DAYS = 14.0
MAX_PER_CREATOR = 2
TOP_N_FINAL = 50
TOP_N_CANDIDATES = 200

# WALS hyperparameters
WALS_FACTORS = 32
WALS_REG = 0.05
WALS_ALPHA = 40
WALS_ITERS = 20


def _timer(label):
    """Context-manager-ish stopwatch returned as a closure."""
    start = time.perf_counter()
    def done(extra=""):
        print(f"  [{label}] {time.perf_counter() - start:5.2f}s {extra}")
    return done


def load_interactions(engine):
    """Build the sparse user×video score matrix.

    Score formula (negatives clipped to 0):
        watched*1.0 + (watch_count>1)*0.5 + thumbs_up*2.0
        + thumbs_down*-3.0 + sent_audio_to_creator*1.5
    """
    sql = text("""
        WITH feed_scores AS (
            SELECT
                f.to_user_id                                      AS user_id,
                f.video_id,
                SUM(
                    (CASE WHEN f.watched_at IS NOT NULL THEN 1.0 ELSE 0 END)
                    + (CASE WHEN COALESCE(f.watch_count, 0) > 1 THEN 0.5 ELSE 0 END)
                    + (CASE WHEN f.reaction = 'thumbs_up'   THEN 2.0  ELSE 0 END)
                    + (CASE WHEN f.reaction = 'thumbs_down' THEN -3.0 ELSE 0 END)
                ) AS s
            FROM feed f
            WHERE f.video_id IS NOT NULL
            GROUP BY f.to_user_id, f.video_id
        ),
        audio_scores AS (
            SELECT
                a.user_id                                         AS user_id,
                a.video_id,
                1.5 * COUNT(*)                                    AS s
            FROM audio a
            WHERE a.video_id IS NOT NULL
              AND COALESCE(a.is_deleted, false) = false
            GROUP BY a.user_id, a.video_id
        ),
        combined AS (
            SELECT user_id, video_id, s FROM feed_scores
            UNION ALL
            SELECT user_id, video_id, s FROM audio_scores
        )
        SELECT user_id, video_id, GREATEST(SUM(s), 0.0) AS score
        FROM combined
        GROUP BY user_id, video_id
        HAVING GREATEST(SUM(s), 0.0) > 0
    """)
    df = pd.read_sql(sql, engine)

    user_ids = pd.Index(df["user_id"].unique(), name="user_id")
    video_ids = pd.Index(df["video_id"].unique(), name="video_id")
    user_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    video_pos = pd.Series(np.arange(len(video_ids)), index=video_ids)

    rows = user_pos.loc[df["user_id"]].to_numpy()
    cols = video_pos.loc[df["video_id"]].to_numpy()
    data = df["score"].to_numpy(dtype=np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(video_ids)))
    return R, list(user_ids), list(video_ids)


def fit_wals(R):
    """Fit WALS on the interaction matrix and return the trained model."""
    model = implicit.als.AlternatingLeastSquares(
        factors=WALS_FACTORS,
        regularization=WALS_REG,
        alpha=WALS_ALPHA,
        iterations=WALS_ITERS,
        use_gpu=False,
    )
    model.fit(R, show_progress=False)
    return model


def generate_candidates(model, R, n=TOP_N_CANDIDATES):
    """Top-N candidates per user from the WALS model.

    Returns dict[user_idx → (video_idx_array, score_array)].
    Already-watched filtering is intentionally deferred to the filter stage.
    """
    cands = {}
    n_users = R.shape[0]
    for u in range(n_users):
        ids, scores = model.recommend(
            u, R[u], N=n, filter_already_liked_items=False
        )
        cands[u] = (np.asarray(ids), np.asarray(scores))
    return cands


def load_filter_data(engine, user_ids, video_ids):
    """Pre-load all data needed for the filter stage as dicts of sets."""
    user_set = set(user_ids)
    video_set = set(video_ids)

    # 1. Global: deleted_or_private videos
    deleted_or_private = set(pd.read_sql(text("""
        SELECT id FROM video
        WHERE COALESCE(is_deleted, false) = true
           OR COALESCE(is_public, true) = false
    """), engine)["id"])

    # 2. Per-user: blocked creators
    blocked_df = pd.read_sql(text("""
        SELECT user_id, to_user_id AS blocked_user
        FROM contact
        WHERE is_blocked = true
          AND to_user_id IS NOT NULL
    """), engine)
    blocked_creators = defaultdict(set)
    for r in blocked_df.itertuples(index=False):
        if r.user_id in user_set:
            blocked_creators[r.user_id].add(r.blocked_user)

    # 3. Per-user: already-watched videos
    watched_df = pd.read_sql(text("""
        SELECT to_user_id AS user_id, video_id
        FROM feed
        WHERE watched_at IS NOT NULL AND video_id IS NOT NULL
    """), engine)
    watched = defaultdict(set)
    for r in watched_df.itertuples(index=False):
        watched[r.user_id].add(r.video_id)

    # 4. Per-user: user-reported videos
    reported_df = pd.read_sql(text("""
        SELECT to_user_id AS user_id, video_id
        FROM feed
        WHERE reaction = 'reported' AND video_id IS NOT NULL
    """), engine)
    reported = defaultdict(set)
    for r in reported_df.itertuples(index=False):
        reported[r.user_id].add(r.video_id)

    # 5. Video metadata for re-rank stage (creator + created_at)
    meta_df = pd.read_sql(text("""
        SELECT id AS video_id, user_id AS creator_id, created_at
        FROM video
        WHERE COALESCE(is_deleted, false) = false
    """), engine)
    meta_df = meta_df[meta_df["video_id"].isin(video_set)]
    video_meta = meta_df.set_index("video_id")

    return {
        "deleted_or_private": deleted_or_private,
        "blocked_creators": blocked_creators,
        "watched": watched,
        "reported": reported,
        "video_meta": video_meta,
    }


def filter_candidates(cands, user_ids, video_ids, fd):
    """Apply hard eligibility filters per user.

    Returns dict[user_idx → (video_idx_array, score_array)] after filtering.
    """
    deleted_or_private = fd["deleted_or_private"]
    blocked_creators = fd["blocked_creators"]
    watched = fd["watched"]
    reported = fd["reported"]
    creator_by_video = fd["video_meta"]["creator_id"].to_dict()

    filtered = {}
    for u_idx, (vid_idxs, scores) in cands.items():
        user_id = user_ids[u_idx]
        u_blocked = blocked_creators.get(user_id, set())
        u_watched = watched.get(user_id, set())
        u_reported = reported.get(user_id, set())

        keep_mask = np.ones(len(vid_idxs), dtype=bool)
        for i, v_idx in enumerate(vid_idxs):
            video_id = video_ids[v_idx]
            if video_id in deleted_or_private:
                keep_mask[i] = False
            elif video_id in u_watched:
                keep_mask[i] = False
            elif video_id in u_reported:
                keep_mask[i] = False
            elif creator_by_video.get(video_id) in u_blocked:
                keep_mask[i] = False

        filtered[u_idx] = (vid_idxs[keep_mask], scores[keep_mask])
    return filtered


def rerank(filtered, user_ids, video_ids, fd, top_n=TOP_N_FINAL):
    """Apply fixed-coefficient re-rank and creator cap. Return long DataFrame."""
    video_meta = fd["video_meta"]
    creator_by_video = video_meta["creator_id"].to_dict()
    created_at_by_video = video_meta["created_at"].to_dict()

    now = pd.Timestamp.now("UTC").tz_localize(None)
    rows = []

    for u_idx, (vid_idxs, scores) in filtered.items():
        if len(vid_idxs) == 0:
            continue

        max_score = scores.max() if scores.max() > 0 else 1.0
        norm_scores = scores / max_score

        ages_days = np.array([
            max(0.0, (now - pd.Timestamp(created_at_by_video.get(video_ids[v])).tz_localize(None)).total_seconds() / 86400.0)
            if video_ids[v] in created_at_by_video else 365.0
            for v in vid_idxs
        ])
        recency = np.exp(-ages_days / RECENCY_HALF_LIFE_DAYS)

        final = W_CAND * norm_scores + W_RECENCY * recency
        order = np.argsort(-final)

        creator_counts = defaultdict(int)
        kept = []
        for j in order:
            video_id = video_ids[vid_idxs[j]]
            creator = creator_by_video.get(video_id)
            if creator is not None and creator_counts[creator] >= MAX_PER_CREATOR:
                continue
            creator_counts[creator] += 1
            kept.append((video_id, float(final[j])))
            if len(kept) >= top_n:
                break

        user_id = user_ids[u_idx]
        for rank, (video_id, fs) in enumerate(kept, start=1):
            rows.append((user_id, video_id, rank, fs))

    return pd.DataFrame(rows, columns=["user_id", "video_id", "rank", "final_score"])


def main():
    load_dotenv()
    engine = create_engine(os.getenv("DATABASE_URL"))

    print("Stage 1 — Load interactions + fit WALS")
    t = _timer("load_interactions")
    R, user_ids, video_ids = load_interactions(engine)
    t(f"R: {R.shape[0]} users × {R.shape[1]} videos, nnz={R.nnz}")

    t = _timer("fit_wals")
    model = fit_wals(R)
    t(f"factors={WALS_FACTORS}, iters={WALS_ITERS}")

    print("\nStage 2 — Generate candidates")
    t = _timer("generate_candidates")
    cands = generate_candidates(model, R, n=TOP_N_CANDIDATES)
    t(f"{TOP_N_CANDIDATES} per user × {len(cands)} users")

    print("\nStage 3 — Load filter data + filter")
    t = _timer("load_filter_data")
    fd = load_filter_data(engine, user_ids, video_ids)
    t(f"deleted/private={len(fd['deleted_or_private'])}, "
      f"users_with_blocks={len(fd['blocked_creators'])}, "
      f"users_with_watched={len(fd['watched'])}")

    t = _timer("filter_candidates")
    filtered = filter_candidates(cands, user_ids, video_ids, fd)
    avg_remaining = np.mean([len(v[0]) for v in filtered.values()])
    t(f"avg {avg_remaining:.1f} candidates/user after filter")

    print("\nStage 4 — Re-rank")
    t = _timer("rerank")
    recs = rerank(filtered, user_ids, video_ids, fd, top_n=TOP_N_FINAL)
    t(f"{len(recs)} total rows ({recs['user_id'].nunique()} users)")

    out = "recommendations.csv"
    recs.to_csv(out, index=False)
    print(f"\nWrote {out}: {len(recs):,} rows, {recs['user_id'].nunique()} users")


if __name__ == "__main__":
    main()
