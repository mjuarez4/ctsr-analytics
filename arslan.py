import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df_csv = pl.read_csv("./ctsr.csv")

#how many CB are gender related

cb_gender = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_topic_gender_majority"]=="t")
).shape[0]

print(f"cb majority and gender majority:", cb_gender)

#How many total CB have mild, moderate, and severe status of severity based on majority vote?
cb_mild = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_severity_mild_count"] >= 3)
).shape[0]

print(f"cb majority and mild count >= 3:", cb_mild)

cb_moderate = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_severity_moderate_count"] >= 3)
).shape[0]

print(f"cb majority and moderate count >= 3:", cb_moderate)

cb_severe = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_severity_severe_count"] >= 3)
).shape[0]

print(f"cb majority and severe count >= 3:", cb_severe)

#How many CB LGBTQ (gender) have mild, moderate, and severe status of severity based on majority vote?

cb_g_mild = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_topic_gender_majority"] == "t") & (df_csv["c_severity_mild_count"] >= 3)
).shape[0]

print(f"cb majority and gender majority and mild count >=3 :", cb_g_mild)

cb_g_moderate = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_topic_gender_majority"] == "t") & (df_csv["c_severity_moderate_count"] >= 3)
).shape[0]

print(f"cb majority and gender majority and moderate count >= 3:", cb_g_moderate)

cb_g_severe = df_csv.filter(
    (df_csv["c_cyberbullying_majority"] == "t") & (df_csv["c_topic_gender_majority"] == "t") & (df_csv["c_severity_severe_count"] >=3 )
).shape[0]

print(f"cb majority and gender majority and severe count >= 3:", cb_g_severe)

