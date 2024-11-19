import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('../AfterSnap/after_snap_3.csv')

# Identify numerical columns and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Prepare the plot
fig, axes = plt.subplots(nrows=len(numerical_cols) + len(categorical_cols), ncols=1,
                         figsize=(8, 5 * (len(numerical_cols) + len(categorical_cols))))

# Ensure axes is iterable even when there is only one plot
if len(numerical_cols) + len(categorical_cols) == 1:
    axes = [axes]

# Plot numerical variables using scatter plots
for i, col in enumerate(numerical_cols):
    axes[i].scatter(df[col], df['x_offset'])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('x_offset')
    axes[i].set_title(f'Scatter plot of {col} vs x_offset')

# Plot categorical variables using bar charts
for i, col in enumerate(categorical_cols, start=len(numerical_cols)):
    # Calculate the mean x_offset for each category in the categorical column
    category_means = df.groupby(col)['x_offset'].mean()

    axes[i].bar(category_means.index, category_means.values)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Mean x_offset')
    axes[i].set_title(f'Bar chart of {col} vs mean x_offset')

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()
