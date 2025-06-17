import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df_csv = pl.read_csv("./ctsr.csv")

unique_count = (
    df_csv.filter(
        (df_csv["c_cyberbullying_majority"] == "t") & 
        (df_csv["c_topic_gender_majority"] == "t")
    )
    .unique(subset=["s_unit_id"])
    .shape[0]
)

print("Unique count:", unique_count)

