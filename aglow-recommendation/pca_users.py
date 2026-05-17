"""User-level PCA on aglow engagement features.

Pulls per-user counts/rates from feed, audio, contact, and video, then runs
PCA on the standardized matrix and writes a 2D scatter to pca_users.png.
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, exc as sa_exc

warnings.filterwarnings("ignore", category=sa_exc.SAWarning)

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

USER_FEATURES_SQL = """
WITH feed_in AS (
    SELECT
        to_user_id AS user_id,
        COUNT(*) AS shown,
        COUNT(watched_at) AS watched,
        COALESCE(SUM(watch_count), 0) AS total_watches,
        SUM(CASE WHEN reaction = 'thumbs_up' THEN 1 ELSE 0 END) AS thumbs_up,
        SUM(CASE WHEN reaction = 'thumbs_down' THEN 1 ELSE 0 END) AS thumbs_down,
        SUM(CASE WHEN reaction = 'reported' THEN 1 ELSE 0 END) AS reported,
        COUNT(DISTINCT from_user_id) AS distinct_senders
    FROM feed
    GROUP BY to_user_id
),
feed_out AS (
    SELECT from_user_id AS user_id, COUNT(*) AS sent_videos
    FROM feed
    GROUP BY from_user_id
),
audio_in AS (
    SELECT
        to_user_id AS user_id,
        COUNT(*) AS audio_received,
        COUNT(listened_at) AS audio_listened
    FROM audio
    WHERE COALESCE(is_deleted, false) = false
    GROUP BY to_user_id
),
audio_out AS (
    SELECT user_id, COUNT(*) AS audio_sent
    FROM audio
    WHERE COALESCE(is_deleted, false) = false
    GROUP BY user_id
),
contacts AS (
    SELECT
        user_id,
        COUNT(*) AS contacts_total,
        SUM(CASE WHEN to_user_id IS NOT NULL THEN 1 ELSE 0 END) AS contacts_on_platform,
        SUM(CASE WHEN is_blocked THEN 1 ELSE 0 END) AS contacts_blocked,
        COALESCE(SUM(total_messages_sent_to), 0) AS msgs_sent_to_contacts
    FROM contact
    GROUP BY user_id
),
uploads AS (
    SELECT user_id, COUNT(*) AS videos_uploaded
    FROM video
    WHERE COALESCE(is_deleted, false) = false
    GROUP BY user_id
)
SELECT
    p.user_id,
    COALESCE(fi.shown, 0)                  AS shown,
    COALESCE(fi.watched, 0)                AS watched,
    COALESCE(fi.total_watches, 0)          AS total_watches,
    COALESCE(fi.thumbs_up, 0)              AS thumbs_up,
    COALESCE(fi.thumbs_down, 0)            AS thumbs_down,
    COALESCE(fi.reported, 0)               AS reported,
    COALESCE(fi.distinct_senders, 0)       AS distinct_senders,
    COALESCE(fo.sent_videos, 0)            AS sent_videos,
    COALESCE(ai.audio_received, 0)         AS audio_received,
    COALESCE(ai.audio_listened, 0)         AS audio_listened,
    COALESCE(ao.audio_sent, 0)             AS audio_sent,
    COALESCE(c.contacts_total, 0)          AS contacts_total,
    COALESCE(c.contacts_on_platform, 0)    AS contacts_on_platform,
    COALESCE(c.contacts_blocked, 0)        AS contacts_blocked,
    COALESCE(c.msgs_sent_to_contacts, 0)   AS msgs_sent_to_contacts,
    COALESCE(u.videos_uploaded, 0)         AS videos_uploaded
FROM profile p
LEFT JOIN feed_in    fi ON fi.user_id = p.user_id
LEFT JOIN feed_out   fo ON fo.user_id = p.user_id
LEFT JOIN audio_in   ai ON ai.user_id = p.user_id
LEFT JOIN audio_out  ao ON ao.user_id = p.user_id
LEFT JOIN contacts   c  ON c.user_id  = p.user_id
LEFT JOIN uploads    u  ON u.user_id  = p.user_id
"""


def main():
    print("Pulling user features...")
    df = pd.read_sql(USER_FEATURES_SQL, engine)
    print(f"  {len(df)} profiles")

    # Restrict to users with any engagement so PCA isn't dominated by zero rows.
    active = df[
        (df["shown"] > 0)
        | (df["audio_received"] > 0)
        | (df["audio_sent"] > 0)
        | (df["videos_uploaded"] > 0)
        | (df["contacts_on_platform"] > 0)
    ].copy()
    print(f"  {len(active)} active users (any engagement signal)")

    # Engineered ratios — bounded, less collinear than raw counts.
    active["watch_rate"] = active["watched"] / active["shown"].clip(lower=1)
    active["like_rate"] = active["thumbs_up"] / active["watched"].clip(lower=1)
    active["dislike_rate"] = active["thumbs_down"] / active["watched"].clip(lower=1)
    active["audio_listen_rate"] = active["audio_listened"] / active["audio_received"].clip(lower=1)
    active["on_platform_ratio"] = active["contacts_on_platform"] / active["contacts_total"].clip(lower=1)

    # Log-transform heavy-tailed counts; PCA on raw counts is dominated by power users.
    count_cols = [
        "shown", "watched", "total_watches", "thumbs_up", "thumbs_down",
        "distinct_senders", "sent_videos", "audio_received", "audio_listened",
        "audio_sent", "contacts_total", "contacts_on_platform",
        "msgs_sent_to_contacts", "videos_uploaded",
    ]
    for c in count_cols:
        active[f"log_{c}"] = np.log1p(active[c])

    feature_cols = (
        [f"log_{c}" for c in count_cols]
        + ["watch_rate", "like_rate", "dislike_rate", "audio_listen_rate", "on_platform_ratio"]
    )
    X = active[feature_cols].values
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(8, Xs.shape[1]))
    Z = pca.fit_transform(Xs)

    evr = pca.explained_variance_ratio_
    print("\nExplained variance by component:")
    for i, v in enumerate(evr, 1):
        print(f"  PC{i}: {v:6.2%}   (cum {evr[:i].sum():6.2%})")

    loadings = pd.DataFrame(
        pca.components_[:4].T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(4)],
    )
    print("\nTop loadings per PC (|loading| >= 0.20):")
    for pc in loadings.columns:
        s = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
        s = s[s.abs() >= 0.20]
        print(f"\n  {pc}")
        for name, val in s.items():
            print(f"    {name:30s} {val:+.3f}")

    _, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    creator = active["videos_uploaded"] > 0
    ax = axes[0]
    ax.scatter(Z[~creator.values, 0], Z[~creator.values, 1], s=10, alpha=0.4,
               label=f"consumer (n={(~creator).sum()})", color="#4a90e2")
    ax.scatter(Z[creator.values, 0], Z[creator.values, 1], s=18, alpha=0.7,
               label=f"creator (n={creator.sum()})", color="#e2574a")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("Users in PC1–PC2 space")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.bar(range(1, len(evr) + 1), evr, color="#4a90e2", label="per PC")
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), color="#e2574a", marker="o", label="cumulative")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = "pca_users.png"
    plt.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
