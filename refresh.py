# pyright: basic
from contextlib import closing
from pathlib import Path
import duckdb
import polars as pl
import altair as alt
# import matplotlib.pyplot as plt
# import numpy as np

color_range = [
    "#08306B",
    "#2171B5",
    "#C6DBEF",
    "#6BAED6",
    "#d9d9d9",
    "#969696",
    "#525252",
    "#000000",
]

heat_range = [
    # "#08306B",
    "#2171B5",
    "#6BAED6",
    "#C6DBEF",
    "#f0f0f0",
]


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
            pl.col("mean_severity").alias("Mean Severity"),
        )
    )

    only_cb_results = (
        time_line.join(severity, on="comment_id", how="inner")
        .filter(pl.col("severity") > 0.0)
        .select(pl.col("comment_number", "severity"))
        .group_by(pl.col("comment_number"))
        .agg(
            pl.col("severity").mean().alias("mean_severity"),
        )
        .sort("comment_number")
        .select(
            pl.col("comment_number"),
            pl.col("mean_severity").alias("Mean Bullying Severity"),
        )
    )
    joined_results = results.join(only_cb_results, on="comment_number", how="inner")
    unpivoted_resutls = joined_results.unpivot(
        on=["Mean Severity", "Mean Bullying Severity"], index="comment_number"
    ).sort("comment_number")

    time_series = (
        alt.Chart(unpivoted_resutls)
        .mark_line()  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("comment_number", scale=alt.Scale(domain=[0, 152])).title(
                "Comment Sequence"
            ),
            y=alt.Y("value", scale=alt.Scale(domain=[0, 2])).title("Severity"),
            strokeDash=alt.StrokeDash("variable").legend(None),
            color=alt.Color(
                "variable",
                scale=alt.Scale(
                    domain=["Mean Severity", "Mean Bullying Severity"],
                    range=["#191717", "#7D7C7C"],
                ),
            ).legend(
                title=None,
                orient="none",
                legendX=290,
                legendY=-5,
                direction="horizontal",
                titleOrient="left",
            ),
        )
    )
    (
        time_series.properties(  # pyright: ignore[reportUnknownMemberType]
            width=600,
            height=75,
        )
        .configure_axis(grid=False, titleFontSize=10)
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
            y=alt.Y("remaped_role", title=None).sort("-x"),
            color=alt.value("#636363"),
        )
    )
    text = bar.mark_text(
        align="center", baseline="bottom", dx=18, dy=5, fontSize=10
    ).encode(text=alt.Text("role_count", format=","), color=alt.value("black"))

    (
        (bar + text)
        .properties(width=200, height=200)
        .configure_axis(grid=False, titleFontSize=10)
        .configure_view(stroke=None)
        .save("role_bar.pdf", format="pdf")
    )

    time_line_role = time_line.join(filtered_roles, on="comment_id", how="inner")

    comment_role_counts = (
        time_line_role.group_by(["comment_number", "remaped_role"])
        .len("role_count")
        .sort("comment_number")
    )

    upper = (
        alt.Chart(comment_role_counts)
        .mark_line(strokeWidth=1.5, fillOpacity=1.0, strokeOpacity=1)  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("comment_number", scale=alt.Scale(domain=[0, 152])).title(None),
            y=alt.Y("role_count", title="Comment Count by Role"),
            color=alt.Color(
                "remaped_role",
                scale=alt.Scale(
                    range=color_range,
                    reverse=True,
                ),
            ).legend(
                title=None,
                orient="none",
                legendX=400,
                # legendY=140,
                # symbolLimit=5,
                columns=2,
                legendY=-15,
                titleOrient="left",
                direction="horizontal",
            ),
        )
        .properties(width=600, height=75)
    )

    lower = (
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
                "remaped_role", scale=alt.Scale(range=color_range, reverse=True)
            ),
        )
        .properties(width=600, height=75)
    )
    (
        alt.vconcat(upper, lower)
        .configure_legend(symbolStrokeColor="black", symbolStrokeWidth=1)
        .configure_axis(grid=False, titleFontSize=10)
        .configure_view(stroke=None)
        .save("combined_time_series_role_percent.pdf", format="pdf")
    )


def plot_topic_heat_map(
    comment_topics: pl.DataFrame, comment_annotations: pl.DataFrame
):
    severity = comment_annotations.select(
        pl.col("comment_id", "assignment_id", "bullying_severity")
    ).filter(pl.col("bullying_severity").is_not_null())

    severity_topic = comment_topics.join(
        severity,
        on=["comment_id", "assignment_id"],
        how="inner",
    ).select(
        pl.col("bullying_severity").str.to_titlecase().alias("bullying_severity"),
        pl.when(pl.col("topic") == "none")
        .then(pl.lit("other"))
        .otherwise(pl.col("topic"))
        .str.replace("_", " ")
        .str.to_titlecase()
        .alias("topic"),
    )
    heat_map = (
        severity_topic.group_by(["topic", "bullying_severity"])
        .agg(pl.len().alias("count"))
        .sort("count")
    )
    figure = (
        alt.Chart(heat_map)
        .mark_rect()
        .encode(
            x=alt.X(
                "bullying_severity:O",
                axis=alt.Axis(labelAngle=0),
                title="Severity",
            ),
            y=alt.Y("topic:O", title="Bullying Topic"),
            color=alt.Color(
                "count:Q",
                title="Count",
                scale=alt.Scale(range=heat_range, domain=[0, 12500], reverse=True),
            ),
        )
    )
    text = (
        alt.Chart(heat_map)
        .mark_text(baseline="middle", fontSize=10)
        .encode(
            x="bullying_severity:O",
            y="topic:O",
            text=alt.Text("count:Q", format=","),
            color=alt.value("black"),  # or 'white' depending on your color scale
        )
    )
    (
        (figure + text)
        .properties(height=200, width=200)
        .configure_axis(titleFontSize=10)
        .save("topic_severity_heatmap.pdf", format="pdf")
    )


def plot_cb(comments: pl.DataFrame, comment_annotations: pl.DataFrame):
    time_line = comments.select(
        pl.col("comment_id"),
        pl.col("comment_created_at")
        .rank("ordinal")
        .over("unit_id")
        .alias("comment_number"),
    )

    counts = (
        comment_annotations.select(
            pl.col("comment_id"),
            pl.when(pl.col("is_cyberbullying"))
            .then(pl.lit("Cyberbullying"))
            .otherwise(pl.lit("Non-Cyberbullying"))
            .alias("is_cyberbullying"),
        )
        .group_by(["comment_id", "is_cyberbullying"])
        .len("count")
    )
    print(counts)
    time_line_bullying = (
        time_line.join(counts, on="comment_id", how="inner")
        .group_by(["comment_number", "is_cyberbullying"])
        .agg(pl.col("count").sum().alias("count"))
    )
    (
        alt.Chart(time_line_bullying)
        .mark_line()  # pyright: ignore[reportUnknownMemberType]
        .encode(
            x=alt.X("comment_number", scale=alt.Scale(domain=[0, 152])).title(
                "Comment Sequence"
            ),
            y=alt.Y("count").title("Cyberbullying Count"),
            # strokeDash=alt.StrokeDash("variable").legend(None),
            color=alt.Color(
                "is_cyberbullying",
                scale=alt.Scale(
                    domain=["Cyberbullying", "Non-Cyberbullying"],
                    range=["#191717", "#7D7C7C"],
                ),
            ).legend(
                title=None,
                orient="none",
                legendX=290,
                legendY=10,
                direction="horizontal",
                titleOrient="left",
            ),
        )
        .properties(
            width=600,
            height=75,
        )
        .configure_legend(symbolStrokeColor="black", symbolStrokeWidth=1)
        .configure_axis(grid=False, titleFontSize=10)
        .configure_view(stroke=None)
        .save("time_series_cb.pdf", format="pdf")
    )


comments = query_duckdb("SELECT * FROM instagram.comments;")
comment_annotations = query_duckdb("SELECT * FROM mturk.comment_annotations;")
comment_topics = query_duckdb("SELECT * FROM mturk.comment_topics;")
# sessions = query_duckdb("SELECT * FROM instagram.sessions;")
# count_sessions(sessions)
# count_comments(comments)
# percent_bully_annotations(comment_annotations)
# count_comments_majority_bullying(comment_annotations)
plot_severity_timeseries(comments, comment_annotations)
plot_role_timeseries(comments, comment_annotations)
plot_topic_heat_map(comment_topics, comment_annotations)
plot_cb(comments, comment_annotations)
