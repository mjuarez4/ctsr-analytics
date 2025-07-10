# pyright: basic
from contextlib import closing
from pathlib import Path
import duckdb
import polars as pl

SAMPLES = 15_000


def query_duckdb(query: str) -> pl.DataFrame:
    assert "SELECT" in query
    DATABASE = Path("./ctsr.duckdb")
    with closing(duckdb.connect(DATABASE, read_only=True)) as con:  # pyright: ignore[reportUnknownMemberType]
        frame = con.execute(query).pl()
    assert isinstance(frame, pl.DataFrame)
    return frame


def write_role_annotation_count_table(comment_annotations: pl.DataFrame):
    role_expression = (
        pl.when(pl.col("bullying_role") == "non_aggressive_victim")
        .then(pl.lit("Non-Agg Victim"))
        .when(pl.col("bullying_role") == "bully_assistant")
        .then(pl.lit("Bully Assist"))
        .when(pl.col("bullying_role") == "passive_bystander")
        .then(pl.lit("Bystander"))
        .when(pl.col("bullying_role") == "bully")
        .then(pl.lit("Bully"))
        .when(pl.col("bullying_role") == "aggressive_defender")
        .then(pl.lit("Agg Defender"))
        .when(pl.col("bullying_role") == "non_aggressive_defender")
        .then(pl.lit("Non-Agg Defender"))
        .when(pl.col("bullying_role") == "aggressive_victim")
        .then(pl.lit("Agg Victim"))
        .otherwise(pl.lit("Unknown"))
        .alias("bullying_role")
    )

    (
        comment_annotations.filter(pl.col("bullying_role").is_not_null())
        .select(role_expression)
        .group_by("bullying_role")
        .len("role_count")
        .sort("role_count", descending=True)
        .write_csv("role_counts.txt")
    )


def write_severity_tables(
    comments: pl.DataFrame, comment_annotations: pl.DataFrame, with_nulls: bool = False
) -> None:
    if with_nulls:
        min_severity = -1.0
    else:
        min_severity = 0.0

    numeric_severity = comment_annotations.select(
        pl.col("comment_id"),
        pl.when(pl.col("bullying_severity") == "mild")
        .then(pl.lit(1.0))
        .when(pl.col("bullying_severity") == "moderate")
        .then(pl.lit(2.0))
        .when(pl.col("bullying_severity") == "severe")
        .then(pl.lit(3.0))
        .when(pl.col("bullying_severity").is_null())
        .then(pl.lit(0.0))
        .alias("severity"),
    ).filter(pl.col("severity") > min_severity)

    comment_sequence = comments.select(pl.col("comment_buckets", "comment_id"))
    severity = numeric_severity.join(comment_sequence, on="comment_id")
    samples: list[pl.DataFrame] = []
    for comment_buckets in range(1, 151):
        this_comment = severity.filter(pl.col("comment_buckets") == comment_buckets)
        for _ in range(1000):
            comment_mean = (
                this_comment.sample(1000, shuffle=True, with_replacement=True)
                .group_by("comment_buckets")
                .agg(pl.col("severity").mean().alias("mean_severity"))
            )
            samples.append(comment_mean)
    sampled_severity = pl.concat(samples, how="vertical")
    bootstrapped_severity_stats = (
        sampled_severity.group_by("comment_buckets")
        .agg(
            pl.col("mean_severity").quantile(0.05).alias("lower_5th"),
            pl.col("mean_severity").mean().alias("mean_severity"),
            pl.col("mean_severity").quantile(0.95).alias("upper_95th"),
        )
        .sort("comment_buckets")
    )
    print(bootstrapped_severity_stats)
    bootstrapped_severity_stats.write_csv(
        f"boot_strap_severity_stats_with_null_{with_nulls}.txt"
    )


def write_role_tables(
    comments: pl.DataFrame, comment_annotations: pl.DataFrame
) -> None:
    role_majority = comment_annotations.group_by(["comment_id", "bullying_role"]).len(
        "role_count"
    )
    top_role = role_majority.group_by("comment_id").agg(
        pl.col("role_count").max().alias("max_role_count")
    )
    filtered_roles = (
        role_majority.join(top_role, on="comment_id", how="inner")
        .filter(pl.col("role_count") >= pl.col("max_role_count"))
        .with_columns(
            pl.when(pl.col("max_role_count") <= 2)
            .then(pl.lit("inconclusive"))
            .otherwise(pl.col("bullying_role"))
            .alias("remaped_role")
        )
        .unique(subset=["comment_id", "remaped_role", "role_count"])
        .select(
            pl.col(
                "comment_id",
                "remaped_role",
            )
        )
        .sort("comment_id")
    )
    comment_sequence = comments.select(pl.col("comment_buckets", "comment_id"))
    time_line_role = filtered_roles.join(comment_sequence, on="comment_id")
    comment_role_counts = (
        (
            time_line_role.group_by(["comment_buckets", "remaped_role"])
            .len("role_count")
            .sort("comment_buckets")
        )
        .pivot("remaped_role", index="comment_buckets", values="role_count")
        .fill_null(0)
    )
    comment_role_counts.write_csv("time_series_role_counts.txt")


def write_topic_severity_heat_map_tables(
    comment_topics: pl.DataFrame, comment_annotations: pl.DataFrame
):
    cleaned_topics = comment_topics.select(
        pl.col("comment_id", "assignment_id"),
        pl.when(pl.col("topic") == "none")
        .then(pl.lit("other"))
        .otherwise(pl.col("topic"))
        .str.replace("_", " ")
        .str.to_titlecase()
        .alias("topic"),
    )
    cleaned_topics.group_by("topic").len("topic_count").sort(
        "topic_count", descending=True
    ).write_csv("topic_counts.txt")

    severity = comment_annotations.select(
        pl.col("comment_id", "assignment_id"),
        pl.col("bullying_severity").str.to_titlecase().alias("bullying_severity"),
    ).filter(pl.col("bullying_severity").is_not_null())

    severity.group_by("bullying_severity").len("severity_count").sort(
        "severity_count", descending=True
    ).write_csv("severity_counts.txt")

    severity_topic = cleaned_topics.join(
        severity,
        on=["comment_id", "assignment_id"],
        how="inner",
    )
    heat_map = (
        severity_topic.group_by(["topic", "bullying_severity"])
        .len("count")
        .sort("count")
    )
    heat_map.write_csv("topic_severit_heat_map.txt")


def write_bully_tables(comments: pl.DataFrame, comment_annotations: pl.DataFrame):
    """
    I want to count the percent of annotators within a comment that voted bully,
    then I want to bootstrapped the percentages for each comment in the comment sequence.
    """
    (
        comment_annotations.select(
            pl.when(pl.col("is_cyberbullying"))
            .then(pl.lit("Bully"))
            .otherwise(pl.lit("Non-Bully"))
            .alias("is_cyberbullying"),
        )
        .group_by("is_cyberbullying")
        .len("bullying_count")
        .sort("bullying_count", descending=True)
        .write_csv("bullying_counts.txt")
    )
    comment_percents = (
        comment_annotations.group_by("comment_id", "is_cyberbullying")
        .len("bullying_count")
        .with_columns(
            (
                pl.col("bullying_count") / pl.sum("bullying_count").over("comment_id")
            ).alias("percentage")
        )
        .filter(pl.col("is_cyberbullying"))
        .select(pl.col("comment_id", "percentage"))
        .sort("comment_id")
    )
    comment_sequence = comments.select(pl.col("comment_buckets", "comment_id"))
    samples: list[pl.DataFrame] = []
    percents = comment_percents.join(comment_sequence, on="comment_id")
    for comment_buckets in range(1, 151):
        this_comment = percents.filter(pl.col("comment_buckets") == comment_buckets)
        for _ in range(1000):
            mean_sample = (
                this_comment.sample(1000, shuffle=True, with_replacement=True)
                .group_by("comment_buckets")
                .agg(pl.col("percentage").mean().alias("mean_percentage"))
            )
            samples.append(mean_sample)
    sampled_percents = pl.concat(samples, how="vertical")
    bootstrapped_percent_stats = (
        sampled_percents.group_by("comment_buckets")
        .agg(
            pl.col("mean_percentage").quantile(0.05).alias("lower_5th"),
            pl.col("mean_percentage").mean().alias("mean").alias("mean_cb_percentage"),
            pl.col("mean_percentage").quantile(0.95).alias("upper_95th"),
        )
        .sort("comment_buckets")
    )
    print(bootstrapped_percent_stats)
    bootstrapped_percent_stats.write_csv("boot_strap_percent_stats.txt")


comments = query_duckdb("SELECT * FROM sequenced_comments;")
comment_annotations = query_duckdb("SELECT * FROM mturk.comment_annotations;")
comment_topics = query_duckdb("SELECT * FROM mturk.comment_topics;")
write_severity_tables(comments, comment_annotations, False)
write_bully_tables(comments, comment_annotations)
write_role_tables(comments, comment_annotations)
write_topic_severity_heat_map_tables(comment_topics, comment_annotations)
write_role_annotation_count_table(comment_annotations)
