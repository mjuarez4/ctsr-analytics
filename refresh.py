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
        .then(pl.lit(1.0))
        .when(pl.col("bullying_severity") == "moderate")
        .then(pl.lit(2.0))
        .when(pl.col("bullying_severity") == "severe")
        .then(pl.lit(3.0))
        .when(pl.col("bullying_severity").is_null())
        .then(pl.lit(0.0))
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
            pl.col("mean_severity").rolling_mean(window_size=5).alias("Mean Severity"),
            pl.col("std_severity").rolling_mean(window_size=5).alias("STD Severity"),
        )
    )
    unpivoted_resutls = results.unpivot(
        on=["Mean Severity", "STD Severity"], index="comment_number"
    ).sort("comment_number")

    time_series = (
        alt.Chart(unpivoted_resutls)
        .mark_line()  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("comment_number", scale=alt.Scale(domain=[0, 152])).title(
                "Comment Sequence"
            ),
            y=alt.Y("value", scale=alt.Scale(domain=[0, 1])).title(None),
            strokeDash=alt.StrokeDash("variable").legend(None),
            color=alt.Color(
                "variable",
                scale=alt.Scale(
                    domain=["Mean Severity", "STD Severity"],
                    range=["#191717", "#7D7C7C"],
                ),
            ).legend(
                title=None,
                orient="none",
                legendX=490,
                legendY=10,
                direction="horizontal",
                titleOrient="left",
            ),
        )
        .properties(
            width=600,
            height=200,
        )
    )
    boxplot = (
        alt.Chart(severity)  # pyright: ignore[reportUnknownMemberType]
        .mark_boxplot(color="#454545", outliers={"size": 5})
        .encode(y=alt.Y("severity", title="Severity"))
        .properties(
            height=200,
        )
    )
    (
        alt.hconcat(boxplot, time_series)  # pyright: ignore[reportUnknownMemberType]
        .properties(title="Temporal Dynamics of Comment Severity")
        .configure_title(fontSize=12, anchor="middle", color="black")
        .configure_axis(grid=False)
        .configure_view(stroke=None)
        .save("time_series_severity.pdf", format="pdf")
    )


def plot_role_timeseries(
    comments: pl.DataFrame, comment_annotations: pl.DataFrame
) -> None:
    time_line = comments.select(
        pl.col("comment_id"),
        pl.col("comment_created_at")
        .rank("ordinal")
        .over("unit_id")
        .alias("comment_number"),
    )
    role_majority = (
        comment_annotations.select(
            pl.col("comment_id"),
            pl.when(pl.col("bullying_role") == "non_aggressive_victim")
            .then(pl.lit("Non-Agg Victim"))
            .when(pl.col("bullying_role") == "aggressive_victim")
            .then(pl.lit("Agg Victim"))
            .when(pl.col("bullying_role") == "non_aggressive_defender")
            .then(pl.lit("Non-Agg Defend"))
            .when(pl.col("bullying_role") == "aggressive_defender")
            .then(pl.lit("Agg Defend"))
            .when(pl.col("bullying_role") == "bully")
            .then(pl.lit("Bully"))
            .when(pl.col("bullying_role") == "bully_assistant")
            .then(pl.lit("Bully Assist"))
            .when(pl.col("bullying_role") == "passive_bystander")
            .then(pl.lit("Bystander"))
            .alias("bullying_role"),
        )
        .group_by(["comment_id", "bullying_role"])
        .len("role_count")
    )
    top_role = role_majority.group_by("comment_id").agg(
        pl.col("role_count").max().alias("max_role_count")
    )
    filtered_roles = (
        (
            role_majority.join(top_role, on="comment_id", how="inner")
            .filter(pl.col("role_count") >= pl.col("max_role_count"))
            .with_columns(
                pl.when(pl.col("max_role_count") <= 2)
                .then(pl.lit("Inconclusive"))
                .otherwise(pl.col("bullying_role"))
                .alias("remaped_role")
            )
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

    bar = (
        alt.Chart(
            filtered_roles.group_by("remaped_role").len("role_count").sort("role_count")
        )
        .mark_bar()  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("role_count", title="Comment Count"),
            y=alt.Y("remaped_role", title="Bullying Role").sort("-x"),
            color=alt.Color(
                "remaped_role", scale=alt.Scale(scheme="category20c", reverse=True)
            ).legend(None),
        )
    )
    text = bar.mark_text(align="center", baseline="bottom", dx=16, dy=5).encode(
        text="role_count", color=alt.value("black")
    )

    (
        (bar + text)
        .configure_axis(grid=False)
        .configure_view(stroke=None)
        .save("role_bar.pdf", format="pdf")
    )

    time_line_role = time_line.join(filtered_roles, on="comment_id", how="inner")
    comment_role_counts = (
        time_line_role.group_by(["comment_number", "remaped_role"])
        .len("role_count")
        .sort("comment_number")
    )
    (
        alt.Chart(comment_role_counts)
        .mark_area()  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("comment_number", scale=alt.Scale(domain=[0, 152])).title(
                "Comment Sequence"
            ),
            y=alt.Y("role_count", title="Comment Comment by Role"),
            color=alt.Color(
                "remaped_role", scale=alt.Scale(scheme="category20c", reverse=True)
            ).legend(
                title=None,
                orient="none",
                legendX=500,
                legendY=10,
                titleOrient="left",
                # direction="horizontal",
            ),
        )
        .configure_legend(symbolStrokeColor="black", symbolStrokeWidth=1)
        .properties(title="Temporal Dynamics of Roles", width=600, height=200)
        .configure_title(fontSize=12, anchor="middle", color="black")
        .configure_axis(grid=False)
        .configure_view(stroke=None)
        .save("time_series_role_counts.pdf", format="pdf")
    )

    (
        alt.Chart(comment_role_counts)
        .mark_area()  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("comment_number", scale=alt.Scale(domain=[0, 152])).title(
                "Comment Sequence"
            ),
            y=alt.Y("role_count", title="Percent of Comments w/ Role").stack(
                "normalize"
            ),
            color=alt.Color(
                "remaped_role", scale=alt.Scale(scheme="category20c", reverse=True)
            ).legend(
                title=None,
                orient="none",
                legendX=500,
                legendY=50,
                titleOrient="left",
                # direction="horizontal",
            ),
        )
        .configure_legend(symbolStrokeColor="black", symbolStrokeWidth=1)
        .properties(title="Temporal Dynamics of Roles", width=600, height=200)
        .configure_title(fontSize=12, anchor="middle", color="black")
        .configure_axis(grid=False)
        .configure_view(stroke=None)
        .save("time_series_role_percent.pdf", format="pdf")
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
plot_role_timeseries(comments, comment_annotations)
