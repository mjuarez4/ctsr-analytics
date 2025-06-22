from contextlib import closing
from pathlib import Path
import duckdb
import polars as pl
import altair as alt
# import matplotlib.pyplot as plt
# import numpy as np


def query_duckdb(query: str) -> pl.DataFrame:
    assert "SELECT" in query
    DATABASE = Path("./ctsr.duckdb")
    with closing(duckdb.connect(DATABASE, read_only=True)) as con:  # pyright: ignore[reportUnknownMemberType]
        frame = con.execute(query).pl()
    assert isinstance(frame, pl.DataFrame)
    return frame


def count_comments(comments: pl.DataFrame) -> None:
    """Compute the total number of comments."""
    print(comments.select(pl.count("comment_id")))


def count_sessions(sessions: pl.DataFrame) -> None:
    """
    Compute the total number of sessions.

    Note we don't need to count the number of sessions without images
    since the sessions table only contains sessions with images.
    """
    print(sessions.select(pl.count("unit_id")))


def count_comments_majority_bullying(comment_annotations: pl.DataFrame) -> None:
    """
    Compute the total numebr of comments where a majority of the annotators agreed on CB.
    """
    result = (
        comment_annotations.select(pl.col("comment_id"), pl.col("is_cyberbullying"))
        .group_by(pl.col("comment_id"), pl.col("is_cyberbullying"))
        .len()
        .filter(pl.col("is_cyberbullying"), pl.col("len") >= 3)
    )
    print(result.count())


def percent_bully_annotations(comment_annotations: pl.DataFrame) -> None:
    """
    Compute the average numebr of annotators who voted bullying on any given comment.
    """
    result = (
        comment_annotations.select(pl.col("comment_id"), pl.col("is_cyberbullying"))
        .group_by(pl.col("comment_id"), pl.col("is_cyberbullying"))
        .len()
        .filter(pl.col("is_cyberbullying"))
        .select(percent_bully=pl.col("len") / 5)
    )

    print(result.mean())


def plot_severity_timeseries(
    comments: pl.DataFrame, comment_annotations: pl.DataFrame
) -> None:
    time_line = comments.select(
        pl.col("comment_id"),
        pl.col("comment_created_at")
        .rank("ordinal")
        .over("unit_id")
        .alias("comment_number"),
    )
    severity = comment_annotations.select(
        pl.col("comment_id"),
        pl.when(pl.col("bullying_severity") == "mild")
        .then(1.0)
        .when(pl.col("bullying_severity") == "moderate")
        .then(2.0)
        .when(pl.col("bullying_severity") == "severe")
        .then(3.0)
        .when(pl.col("bullying_severity").is_null())
        .then(0.0)
        .alias("severity"),
    )
    results = (
        time_line.join(severity, on="comment_id", how="inner")
        .select(pl.col("comment_number", "severity"))
        .group_by(pl.col("comment_number"))
        .agg(
            pl.col("severity").mean().alias("mean_severity"),
            pl.col("severity").std().alias("std_severity"),
        )
        .sort("comment_number")
        .select(
            pl.col("comment_number"),
            pl.col("mean_severity").rolling_mean(window_size=5).alias("mean_severity"),
            pl.col("std_severity").rolling_mean(window_size=5).alias("std_severity"),
        )
    )
    unpivoted_resutls = results.unpivot(
        on=["mean_severity", "std_severity"], index="comment_number"
    ).sort("comment_number")

    time_series = (
        alt.Chart(unpivoted_resutls)
        .mark_line()
        .encode(
            x=alt.X("comment_number").title("Comment Sequence"),
            y=alt.Y("value").title(None),
            color=alt.Color(
                "variable",
                scale=alt.Scale(
                    domain=["mean_severity", "std_severity"],
                    range=["#191717", "#7D7C7C"],
                ),
            ).legend(None),
        )
        .properties(
            width=600,
            height=200,
        )
    )
    boxplot = (
        alt.Chart(severity)
        .mark_boxplot(color="#454545")
        .encode(y=alt.Y("severity", title="Severity"))
        .properties(
            height=200,
        )
    )
    (
        alt.hconcat(boxplot, time_series)
        .properties(title="testing")
        .configure_title(fontSize=20, color="black")
        # .configure_axis(grid=False)
        # .configure_view(stroke=None)
        .save("testing.pdf", format="pdf")
    )


comments = query_duckdb("SELECT * FROM instagram.comments;")
comment_annotations = query_duckdb("SELECT * FROM mturk.comment_annotations;")
# comment_topics = query_duckdb("SELECT * FROM mturk.comment_topics;")
# sessions = query_duckdb("SELECT * FROM instagram.sessions;")
# count_sessions(sessions)
# count_comments(comments)
# percent_bully_annotations(comment_annotations)
# count_comments_majority_bullying(comment_annotations)
plot_severity_timeseries(comments, comment_annotations)
