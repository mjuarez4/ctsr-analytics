
#used for analyzing the ctsr dataset

import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df_csv = pl.read_csv("./ctsr.csv")


#counting total number of cb post

#unique by unit id
# has_images in the new dataset
# where number_of_bullying_annotations >= 3
total_cb_posts = df_csv.filter(df_csv["c_cyberbullying_majority"] == "t").shape[0]
tot_noncb_posts = df_csv.filter(df_csv["c_cyberbullying_majority"] == "f").shape[0]
print(tot_noncb_posts)
cb1  = df_csv.filter(pl.col("s_number_of_bully_annotations") >= 3)
cb1_comments = cb1.select(
        pl.col("s_unit_id").n_unique().alias("cb post count")
        )
print(cb1_comments)

total_posts = df_csv.shape[0]

tposts = df_csv.select(
        pl.col("s_unit_id").n_unique().alias("total post session")
        )
print(tposts)

#noncb
noncb = tposts - cb1_comments
print(noncb)

#num of posts with CB and photos
cb_photos = cb1.filter(pl.col("s_has_image") == "t").group_by("s_unit_id").agg([
        pl.count().alias("num_cb_posts_w_images")
        ])
print(cb_photos)

#num of posts without cb

cb1_non = df_csv.filter(pl.col("s_number_of_bully_annotations") < 3)
noncb_photos = cb1_non.filter(pl.col("s_has_image") == "t").group_by("s_unit_id").agg([
        pl.count().alias("num_noncb_posts_w_images")
        ])
print(noncb_photos)

#percent of cb posts relative to entire dataset
pc_t = (total_cb_posts / total_posts)

#majority select disability
tot_disability = df_csv.filter(df_csv["c_topic_disability_any"] == "t").shape[0]

#majority select gender
tot_gender = df_csv.filter(df_csv["c_topic_gender_any"] == "t").shape[0]

#majority select intellectual
tot_intellectual = df_csv.filter(df_csv["c_topic_intellectual_any"] == "t").shape[0]
#majority select other
tot_other = df_csv.filter(df_csv["c_topic_other_any"] == "t").shape[0]

#majority select physical
tot_physical = df_csv.filter(df_csv["c_topic_physical_any"] == "t").shape[0]

#majority select political
tot_political = df_csv.filter(df_csv["c_topic_political_any"] == "t").shape[0]

#majority select religious
tot_religious = df_csv.filter(df_csv["c_topic_religious_any"] ==  "t").shape[0]

#majority select race
tot_race = df_csv.filter(df_csv["c_topic_race_any"] == "t").shape[0]

#majority select sexual
tot_sexual = df_csv.filter(df_csv["c_topic_sexual_any"] == "t").shape[0]

#majority select social status
tot_social = df_csv.filter(df_csv["c_topic_social_status_any"] == "t").shape[0]

#3 min, max, avg num of comments
#cb sessions
tot_cb_posts = df_csv.filter(pl.col("s_number_of_bully_annotations") >= 3)
cb_comments = tot_cb_posts.group_by("s_unit_id").agg([
    pl.count("c_comment_content").alias("comment_count")
    ])

summary_cb = cb_comments.select([
    pl.col("comment_count").min().alias("min_comments"),
    pl.col("comment_count").max().alias("max comments"),
    pl.col("comment_count").mean().alias("avg comments")
    ])

print(summary_cb)

#non-cb sessions

tot_noncb_posts = df_csv.filter(pl.col("s_number_of_bully_annotations") < 3)
noncb_comments = tot_noncb_posts.group_by("s_unit_id").agg([
    pl.count("c_comment_content").alias("comment_count")
    ])

summary_noncb = noncb_comments.select([
    pl.col("comment_count").min().alias("min_comments"),
    pl.col("comment_count").max().alias("max comments"),
    pl.col("comment_count").mean().alias("avg comments")
    ])

print(summary_noncb)

#5 comment length
#all sessions
tot_len = df_csv.with_columns(
        pl.col("c_comment_content").str.len_chars().alias("comment_length")
        )

summary_totlen = tot_len.select([ 
    pl.col("comment_length").min().alias("min_length"),
    pl.col("comment_length").max().alias("max length"),
    pl.col("comment_length").mean().alias("avg length")
    ])

print(summary_totlen)

#cb sessions
cb_len = tot_cb_posts.with_columns(
        pl.col("c_comment_content").str.len_chars().alias("comment_length")
        )

summary_cblen = cb_len.select([ 
    pl.col("comment_length").min().alias("min_length"),
    pl.col("comment_length").max().alias("max length"),
    pl.col("comment_length").mean().alias("avg length")
    ])

print(summary_cblen)

#non-cb sessions
noncb_len = tot_noncb_posts.with_columns(
        pl.col("c_comment_content").str.len_chars().alias("comment_length")
        )

summary_noncblen = noncb_len.select([ 
    pl.col("comment_length").min().alias("min_length"),
    pl.col("comment_length").max().alias("max length"),
    pl.col("comment_length").mean().alias("avg length")
    ])

print(summary_noncblen)

print(f"total cb posts:" ,total_cb_posts, total_cb_posts/total_posts)
print(f"total disability posts:",tot_disability, tot_disability/total_posts)
print(f"total gender posts:",tot_gender, tot_gender/total_posts)
print(f"total intellectual posts:",tot_intellectual, tot_intellectual/total_posts)
print(f"total other posts:", tot_other, tot_other/total_posts)
print(f"total physical posts:", tot_physical, tot_physical/total_posts)
print(f"total political posts:", tot_political, tot_political/total_posts)
print(f"total religious posts:", tot_religious, tot_religious/total_posts)
print(f"total race posts:", tot_race, tot_race/total_posts)
print(f"total sexual posts:", tot_sexual, tot_sexual/total_posts)
print(f"total social status posts", tot_social, tot_social/total_posts)
print(f"percent of cb posts",pc_t)


#arslan requests

#num of comments where cyberbullying count >= 1


num_cb_related= df_csv.filter(pl.col("c_cyberbullying_count") >= 1)
num_cb_comments = num_cb_related.select(
        pl.col("c_comment_content").count()
        )
print(num_cb_comments)

#number of comments related to gender where at least one annotator flagged

num_gender_related = df_csv.filter(pl.col("c_topic_gender_any") == "t")
num_gender_comments = num_gender_related.select(
        pl.col("c_comment_content").count()
        )
print(num_gender_comments)

#num related to gender (majority)
num_gender_maj = df_csv.filter(pl.col("c_topic_gender_majority") == "t")
num_gender_comments_maj = num_gender_maj.select(
        pl.col("c_comment_content").count()
        )
print(num_gender_comments_maj)

#histogram

gender_dist = (
        df_csv.filter(df_csv['c_topic_gender_majority'] == 't')
        )


#grouped_gend = (
#        gender_dist.group_by('s_unit_id')
#        .agg([
#            pl.sum('c_severity_mild_count').alias('mild'),
#            pl.sum('c_severity_moderate_count').alias('moderate'),
#            pl.sum('c_severity_severe_count').alias('severe')
#            ])
#        )

#severity_sums = grouped_gend.select([
#    pl.sum('mild').alias('mild'),
#    pl.sum('moderate').alias('moderate'),
#    pl.sum('severe').alias('severe')
#    ])
#
#severity_dict = severity_sums.to_dict(as_series=False)

#counts = [severity_dict['mild'][0],severity_dict['moderate'][0], severity_dict['severe'][0]]



#df_csv
cb_filt = df_csv.filter(df_csv['s_number_of_bully_annotations'] >= 1)

mild_count = (
   cb_filt .filter(cb_filt['c_severity_mild_count'] >= 3)
    .select('c_comment_id')
    .unique()
    .shape[0]  
)

moderate_count = (
   cb_filt .filter(cb_filt['c_severity_moderate_count'] >= 3)
    .select('c_comment_id')
    .unique()
    .shape[0]
)

severe_count = (
   cb_filt .filter(cb_filt['c_severity_severe_count'] >= 3)
    .select('c_comment_id')
    .unique()
    .shape[0]
)

print("Comment counts for each severity level:")
print(f"Mild: {mild_count}")
print(f"Moderate: {moderate_count}")
print(f"Severe: {severe_count}")

#severity_levels = ['mild', 'moderate', 'severe']
#counts = [mild_count, moderate_count, severe_count]

#plt.bar(severity_levels, counts)

#bars = plt.bar(severity_levels, counts, color="steelblue")

# Add value labels on top of bars
#for bar in bars:
#   yval = bar.get_height()
#   plt.text(bar.get_x() + bar.get_width()/2, yval + 2,  # Adjust position slightly above bar
#            str(yval), ha='center', fontsize=10, fontweight='bold')

##plt.bar(severity_levels, counts)
#plt.title('Cyberbullying Related Comments Distribution by Severity Level')
#plt.xlabel('Severity Level')
#plt.ylabel('Number of Comments')
#plt.savefig('cb_dist.png')  
#print("Plot saved as 'unique_unit_id_distribution.png'")


#cb_filt = df_csv.filter(df_csv['c_severity_mild_count'] >= 3)

"""
disability_maj = (
        cb_filt.filter(cb_filt['c_topic_disability_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )


gender_maj= (
        cb_filt.filter(cb_filt['c_topic_gender_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



intellectual_maj= (
        cb_filt.filter(cb_filt['c_topic_intellectual_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



other_maj= (
        cb_filt.filter(cb_filt['c_topic_other_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



physical_maj= (
        cb_filt.filter(cb_filt['c_topic_physical_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



political_maj= (
        cb_filt.filter(cb_filt['c_topic_political_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



race_maj= (
        cb_filt.filter(cb_filt['c_topic_race_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



religious_maj= (
        cb_filt.filter(cb_filt['c_topic_religious_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



sexual_maj= (
        cb_filt.filter(cb_filt['c_topic_sexual_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )



social_stat_maj= (
        cb_filt.filter(cb_filt['c_topic_social_status_majority'] == 't')
        .select('c_comment_id')
        .unique()
        .shape[0]
        )


"""



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


#1 vote
disability_one = df_csv.filter(
    (df_csv["c_topic_disability_count"] == 1)
).shape[0]

gender_one = df_csv.filter(
    (df_csv["c_topic_gender_count"]== 1)
).shape[0]

intellectual_one = df_csv.filter(
    (df_csv["c_topic_intellectual_count"] == 1)
).shape[0]

other_one = df_csv.filter(
    (df_csv["c_topic_other_count"] == 1)
).shape[0]

physical_one = df_csv.filter(
    (df_csv["c_topic_physical_count"] == 1)
).shape[0]

political_one = df_csv.filter(
    (df_csv["c_topic_political_count"] == 1)
).shape[0]

race_one = df_csv.filter(
    (df_csv["c_topic_race_count"] == 1)
).shape[0]

religious_one = df_csv.filter(
    (df_csv["c_topic_religious_count"] == 1)
).shape[0]

sexual_one = df_csv.filter(
    (df_csv["c_topic_sexual_count"] == 1)
).shape[0]
social_stat_one = df_csv.filter(
    (df_csv["c_topic_social_status_count"] == 1)
).shape[0]

#2 votes
disability_two = df_csv.filter(
    (df_csv["c_topic_disability_count"] == 2)
).shape[0]

gender_two = df_csv.filter(
    (df_csv["c_topic_gender_count"] == 2)
).shape[0]

intellectual_two = df_csv.filter(
    (df_csv["c_topic_intellectual_count"] == 2)
).shape[0]

other_two = df_csv.filter(
    (df_csv["c_topic_other_count"] == 2)
).shape[0]

physical_two = df_csv.filter(
    (df_csv["c_topic_physical_count"] == 2)
).shape[0]

political_two = df_csv.filter(
    (df_csv["c_topic_political_count"] == 2)
).shape[0]

race_two = df_csv.filter(
    (df_csv["c_topic_race_count"] == 2)
).shape[0]

religious_two = df_csv.filter(
    (df_csv["c_topic_religious_count"] == 2)
).shape[0]

sexual_two = df_csv.filter(
    (df_csv["c_topic_sexual_count"] == 2)
).shape[0]
social_stat_two = df_csv.filter(
    (df_csv["c_topic_social_status_count"] == 2)
).shape[0]

#three
disability_three = df_csv.filter(
    (df_csv["c_topic_disability_count"] == 3)
).shape[0]

gender_three = df_csv.filter(
    (df_csv["c_topic_gender_count"] == 3)
).shape[0]

intellectual_three = df_csv.filter(
    (df_csv["c_topic_intellectual_count"] == 3)
).shape[0]

other_three = df_csv.filter(
    (df_csv["c_topic_other_count"] == 3)
).shape[0]

physical_three = df_csv.filter(
    (df_csv["c_topic_physical_count"] == 3)
).shape[0]

political_three = df_csv.filter(
    (df_csv["c_topic_political_count"] == 3)
).shape[0]

race_three = df_csv.filter(
    (df_csv["c_topic_race_count"] == 3)
).shape[0]

religious_three = df_csv.filter(
    (df_csv["c_topic_religious_count"] == 3)
).shape[0]

sexual_three = df_csv.filter(
    (df_csv["c_topic_sexual_count"] == 3)
).shape[0]
social_stat_three = df_csv.filter(
    (df_csv["c_topic_social_status_count"] == 3)
).shape[0]

#four
disability_four = df_csv.filter(
    (df_csv["c_topic_disability_count"] == 4)
).shape[0]

gender_four = df_csv.filter(
    (df_csv["c_topic_gender_count"] == 4)
).shape[0]

intellectual_four = df_csv.filter(
    (df_csv["c_topic_intellectual_count"]  == 4)
).shape[0]

other_four = df_csv.filter(
    (df_csv["c_topic_other_count"]  == 4)
).shape[0]

physical_four = df_csv.filter(
    (df_csv["c_topic_physical_count"]  == 4)
).shape[0]

political_four = df_csv.filter(
    (df_csv["c_topic_political_count"]  == 4)
).shape[0]

race_four = df_csv.filter(
    (df_csv["c_topic_race_count"]  == 4)
).shape[0]

religious_four = df_csv.filter(
    (df_csv["c_topic_religious_count"]  == 4)
).shape[0]

sexual_four = df_csv.filter(
    (df_csv["c_topic_sexual_count"]  == 4)
).shape[0]
social_stat_four = df_csv.filter(
    (df_csv["c_topic_social_status_count"]  == 4)
).shape[0]

#five
disability_five = df_csv.filter(
    (df_csv["c_topic_disability_count"] == 5)
).shape[0]

gender_five = df_csv.filter(
    (df_csv["c_topic_gender_count"] == 5)
).shape[0]

intellectual_five = df_csv.filter(
    (df_csv["c_topic_intellectual_count"] == 5)
).shape[0]

other_five = df_csv.filter(
    (df_csv["c_topic_other_count"]== 5)
).shape[0]

physical_five = df_csv.filter(
    (df_csv["c_topic_physical_count"] == 5)
).shape[0]

political_five = df_csv.filter(
    (df_csv["c_topic_political_count"] == 5)
).shape[0]

race_five = df_csv.filter(
    (df_csv["c_topic_race_count"] == 5)
).shape[0]

religious_five = df_csv.filter(
    (df_csv["c_topic_religious_count"]== 5)
).shape[0]

sexual_five = df_csv.filter(
    (df_csv["c_topic_sexual_count"] == 5)
).shape[0]
social_stat_five = df_csv.filter(
    (df_csv["c_topic_social_status_count"]== 5)
).shape[0]
print("What")
print(social_stat_five)

#stacked bar chart

"""
"disability_0" : disability_none,
    "gender_0": gender_none,
    "intellectual_0": intellectual_none,
    "other_0":other_none,
    "physical_0":physical_none,
    "political_0":political_none,
    "race_0":race_none,
    "religious_0":religious_none,
    "sexual_0":sexual_none,
    "social_status_0":social_stat_none,
"""
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
colors = plt.cm.viridis(np.linspace(0, 1, len(score_levels)))

# Create Stacked Bar Chart
fig, ax = plt.subplots(figsize=(12, 7))

bottom_values = np.zeros(len(categories))  # Base for stacking
for i, score in enumerate(score_levels):
    ax.bar(categories, df_wide[score].to_list(), label=f"Score {score}", bottom=bottom_values, color=colors[i])
    bottom_values += np.array(df_wide[score].to_list())

# Labels and title
ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.set_title("Stacked Bar Chart of Categories by Score Level")
ax.legend(title="Score Level")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.savefig('stacked_bar_chart.png')  
print("Plot saved as 'stacked_bar_chart.png'")

plt.show()

#print("Severity Dist for each cyberbullying topic:")
#severity_levels = [
#    "Disab.", "Gender", "Intel.", "Other", "Phys.", 
#    "Pol.", "Race", "Relig.", "Sex.", "Soc. Stat."
#]
#counts = [disability_maj, gender_maj, intellectual_maj, other_maj, physical_maj, political_maj, race_maj, religious_maj, sexual_maj, social_stat_maj]
#bars = plt.bar(severity_levels, counts, color="steelblue")

# Add value labels on top of bars
#for bar in bars:
#   yval = bar.get_height()
#   plt.text(bar.get_x() + bar.get_width()/2, yval + 2,  # Adjust position slightly above bar   
#            str(yval), ha='center', fontsize=10, fontweight='bold')

#plt.bar(severity_levels, counts)
#plt.xticks(rotation=30, ha="right", fontsize=10)
#plt.title('Cyberbullying Disagreement Comments Distribution by Topic')
#plt.xlabel('Topic')
#plt.ylabel('Number of Unique Comments')
#lt.savefig('cb_disagreement.png')  
#print("Plot saved as 'cb_disagreement.png'")


print("Requests from overleaf")

#disagreements over cb / on the fence
disagreement_cb = df_csv.filter(
    (df_csv["c_cyberbullying_count"] >= 2) & (df_csv["c_cyberbullying_count"] <= 3)
).height


two_few = df_csv.filter(
    (df_csv["c_cyberbullying_count"] <= 2)
).height

major_ag = df_csv.filter(
    (df_csv["c_cyberbullying_count"] >= 3)
).height

total_ag = df_csv.filter(
    (df_csv["c_cyberbullying_count"] == 5)
).height


#counts = [disagreement_cb, ]
print(disagreement_cb)
print(two_few)
print(major_ag)
print(total_ag)