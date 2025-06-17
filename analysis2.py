import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df_csv = pl.read_csv("./ctsr.csv")

#same count
same_count = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) | (df_csv["c_topic_disability_count"] == 5)| (df_csv["c_topic_gender_count"] == 5) | (df_csv["c_topic_intellectual_count"] == 5) | (df_csv["c_topic_other_count"] == 5) | (df_csv["c_topic_physical_count"] == 5)| (df_csv["c_topic_political_count"] == 5) | (df_csv["c_topic_race_count"] == 5) | (df_csv["c_topic_religious_count"] == 5) | (df_csv["c_topic_sexual_count"] == 5) | (df_csv["c_topic_social_status_count"] == 5) | (df_csv["c_severity_mild_count"] == 5) | (df_csv["c_severity_moderate_count"] == 5) | (df_csv["c_severity_severe_count"] == 5) | (df_csv["c_role_cb_aggressive_defender_count"] == 5) | (df_csv["c_role_cb_aggressive_victim_role_count"] == 5)| (df_csv["c_role_bully_count"] == 5)| (df_csv["c_role_cb__bully_assistant_count"] == 5)| (df_csv["c_role_noncb_passive_bystander_count"] == 5)| (df_csv["c_role_noncb_non_aggressive_victim_count"] == 5)| (df_csv["c_role_noncb_aggressive_defender_count"] == 5)| (df_csv["c_subrole_noncb_direct_to_the_bully_count"] == 5)| (df_csv["c_subrole_noncb_direct_to_the_bully_count"] == 5)| (df_csv["c_subrole_noncb_support_to_the_victim_count"] == 5)
).shape[0]
#graph
print(f"same count", same_count)

# Filter rows where any of the specified columns equal 5
import polars as pl

# Build the condition for the topic group: exactly one equals 5.
topic_condition = (
    (df_csv["c_topic_disability_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_gender_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_intellectual_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_other_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_physical_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_political_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_race_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_religious_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_sexual_count"] == 5).cast(pl.Int64) +
    (df_csv["c_topic_social_status_count"] == 5).cast(pl.Int64)
) == 1

# Build the condition for the severity group: exactly one equals 5.
severity_condition = (
    (df_csv["c_severity_mild_count"] == 5).cast(pl.Int64) +
    (df_csv["c_severity_moderate_count"] == 5).cast(pl.Int64) +
    (df_csv["c_severity_severe_count"] == 5).cast(pl.Int64)
) == 1

# Now combine with the condition that c_cyberbullying_count is 5.
filtered_entries_cb = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & topic_condition & severity_condition
)

filtered_entries_noncb = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) | (df_csv["c_cyberbullying_count"] == 0)
)
print(f"len non cb",filtered_entries_noncb)


# Save the filtered rows to a CSV file.
filtered_entries_cb.write_csv("filtered_entries_cb.csv")

# Optionally, print out the filtered DataFrame to inspect it:
print(f"len filtered", len(filtered_entries_cb))

testing_maddie1 = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 0)
).shape[0]

testing_maddie2 = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5)
).shape[0]

print(f"maddie1", testing_maddie1)
print(f"maddie2", testing_maddie2)



disability_none = df_csv.filter(
    (df_csv["c_topic_disability_count"] == 0)
).shape[0]

gender_none = df_csv.filter(
    (df_csv["c_topic_gender_count"] == 0)
).shape[0]

intellectual_none = df_csv.filter(
    (df_csv["c_topic_intellectual_count"] == 0)
).shape[0]

other_none = df_csv.filter(
    (df_csv["c_topic_other_count"] == 0)
).shape[0]

physical_none = df_csv.filter(
    (df_csv["c_topic_physical_count"] == 0)
).shape[0]

political_none = df_csv.filter(
    (df_csv["c_topic_political_count"] == 0)
).shape[0]

race_none = df_csv.filter(
    (df_csv["c_topic_race_count"] == 0)
).shape[0]

religious_none = df_csv.filter(
    (df_csv["c_topic_religious_count"] == 0)
).shape[0]

sexual_none = df_csv.filter(
    (df_csv["c_topic_sexual_count"] == 0)
).shape[0]
social_stat_none = df_csv.filter(
    (df_csv["c_topic_social_status_count"] == 0)
).shape[0]

# 1 vote
disability_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_disability_any"] == "t")
).shape[0]

gender_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_gender_any"] == "t")
).shape[0]

intellectual_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_intellectual_any"] == "t")
).shape[0]

other_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_other_any"] == "t")
).shape[0]

physical_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_physical_any"] == "t")
).shape[0]

political_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_political_any"] == "t")
).shape[0]

race_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_race_any"] == "t")
).shape[0]

religious_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_religious_any"] == "t")
).shape[0]

sexual_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_sexual_any"] == "t")
).shape[0]

social_stat_one = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 1) & (df_csv["c_topic_social_status_any"] == "t")
).shape[0]


# 2 votes
disability_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_disability_any"] == "t")
).shape[0]

gender_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_gender_any"] == "t")
).shape[0]

intellectual_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_intellectual_any"] == "t")
).shape[0]

other_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_other_any"] == "t")
).shape[0]

physical_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_physical_any"] == "t")
).shape[0]

political_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_political_any"] == "t")
).shape[0]

race_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_race_any"] == "t")
).shape[0]

religious_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_religious_any"] == "t")
).shape[0]

sexual_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_sexual_any"] == "t")
).shape[0]

social_stat_two = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 2) & (df_csv["c_topic_social_status_any"] == "t")
).shape[0]


# 3 votes
disability_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_disability_any"] == "t")
).shape[0]

gender_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_gender_any"] == "t")
).shape[0]

intellectual_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_intellectual_any"] == "t")
).shape[0]

other_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_other_any"] == "t")
).shape[0]

physical_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_physical_any"] == "t")
).shape[0]

political_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_political_any"] == "t")
).shape[0]

race_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_race_any"] == "t")
).shape[0]

religious_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_religious_any"] == "t")
).shape[0]

sexual_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_sexual_any"] == "t")
).shape[0]

social_stat_three = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 3) & (df_csv["c_topic_social_status_any"] == "t")
).shape[0]


# 4 votes
disability_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_disability_any"] == "t")
).shape[0]

gender_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_gender_any"] == "t")
).shape[0]

intellectual_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_intellectual_any"] == "t")
).shape[0]

other_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_other_any"] == "t")
).shape[0]

physical_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_physical_any"] == "t")
).shape[0]

political_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_political_any"] == "t")
).shape[0]

race_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_race_any"] == "t")
).shape[0]

religious_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_religious_any"] == "t")
).shape[0]

sexual_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_sexual_any"] == "t")
).shape[0]

social_stat_four = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 4) & (df_csv["c_topic_social_status_any"] == "t")
).shape[0]


# 5 votes
disability_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_disability_any"] == "t")
).shape[0]

gender_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_gender_any"] == "t")
).shape[0]

intellectual_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_intellectual_any"] == "t")
).shape[0]

other_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_other_any"] == "t")
).shape[0]

physical_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_physical_any"] == "t")
).shape[0]

political_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_political_any"] == "t")
).shape[0]

race_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_race_any"] == "t")
).shape[0]

religious_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_religious_any"] == "t")
).shape[0]

sexual_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_sexual_any"] == "t")
).shape[0]

social_stat_five = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5) & (df_csv["c_topic_social_status_any"] == "t")
).shape[0]



print("What")
print(social_stat_five)
dataset_new = pl.DataFrame({
    
    "disability_1" : disability_one,
    "gender_1": gender_one,
    "intellectual_1": intellectual_one,
    "other_1": other_one,
    "physical_1": physical_one,
    "political_1": political_one,
    "race_1": race_one,
    "religious_1": religious_one,
    "sexual_1": sexual_one,
    "social_status_1": social_stat_one,

    "disability_2" : disability_two,
    "gender_2": gender_two,
    "intellectual_2": intellectual_two,
    "other_2": other_two,
    "physical_2": physical_two,
    "political_2": political_two,
    "race_2": race_two,
    "religious_2": religious_two,
    "sexual_2": sexual_two,
    "social_status_2": social_stat_two,

    "disability_3" : disability_three,
    "gender_3": gender_three,
    "intellectual_3": intellectual_three,
    "other_3": other_three,
    "physical_3": physical_three,
    "political_3": political_three,
    "race_3": race_three,
    "religious_3": religious_three,
    "sexual_3": sexual_three,
    "social_status_3": social_stat_three,

    "disability_4" : disability_four,
    "gender_4": gender_four,
    "intellectual_4": intellectual_four,
    "other_4": other_four,
    "physical_4": physical_four,
    "political_4": political_four,
    "race_4": race_four,
    "religious_4": religious_four,
    "sexual_4": sexual_four,
    "social_status_4": social_stat_four,

    "disability_5" : disability_five,
    "gender_5": gender_five,
    "intellectual_5": intellectual_five,
    "other_5": other_five,
    "physical_5": physical_five,
    "political_5": political_five,
    "race_5": race_five,
    "religious_5": religious_five,
    "sexual_5": sexual_five,
    "social_status_5": social_stat_five
})


print(dataset_new)

# Define categories
categories = ["disability", "gender", "intellectual", "other", "physical", 
              "political", "race", "religious", "sexual", "social_status"]

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Use `unpivot` instead of deprecated `melt`
dataset_long = dataset_new.unpivot(
    index=[],  # No explicit index
    on=dataset_new.columns,  # Columns to transform
    variable_name="Category_Score", 
    value_name="Value"
)

# Debug: Check dataset structure after unpivoting
print("After unpivoting:", dataset_long.head(10))

# Extract Category and Score Level correctly
dataset_long = dataset_long.with_columns(
    dataset_long["Category_Score"]
    .str.extract_groups(r"(.+)_([0-9]+)")
    .alias("Category_Score_Struct")
).unnest("Category_Score_Struct")

# Debug: Print schema to check column names
print("Schema after extracting:", dataset_long.schema)

# Rename extracted columns (if they appear as "1" and "2" instead of expected names)
dataset_long = dataset_long.rename({"1": "Category", "2": "Score_Level"})

# Ensure `Score_Level` is an integer
dataset_long = dataset_long.with_columns(
    dataset_long["Score_Level"].cast(pl.Int64)
).drop("Category_Score")

# **Sort by Score Level and Category**
dataset_long = dataset_long.sort(["Score_Level", "Category"], descending=[False, False])

# Debugging
print("Final Processed Data (Sorted):", dataset_long.head(10))

# Group by Category and Score Level, summing values
dataset_pivot = dataset_long.group_by(["Category", "Score_Level"]).agg(pl.sum("Value"))

# Pivot the data using the new correct argument `on` instead of `columns`
df_wide = dataset_pivot.pivot(index="Category", on="Score_Level", values="Value").fill_null(0)

# Extract categories for x-axis
categories = df_wide["Category"].to_list()
score_levels = sorted([col for col in df_wide.columns if col != "Category"], key=int)  # Ensure numeric sorting

# Define colors
#colors = plt.cm.viridis(np.linspace(0, 1, len(score_levels)))
colors = plt.cm.plasma(np.linspace(0, 1, len(score_levels)))
# Create Stacked Bar Chart
fig, ax = plt.subplots(figsize=(12, 7))

bottom_values = np.zeros(len(categories))  # Base for stacking
for i, score in enumerate(score_levels):
    ax.bar(categories, df_wide[score].to_list(), label=f"Cyberbullying Votes {score}", bottom=bottom_values, color=colors[i])
    bottom_values += np.array(df_wide[score].to_list())

# Labels and title
ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.set_title("Stacked Bar Chart of Categories by Num of Cyberbullying Votes")
ax.legend(title="Num of Cyberbullying Votes")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.savefig('stacked_bar_chart.png')  
print("Plot saved as 'stacked_bar_chart.png'")

plt.show()

