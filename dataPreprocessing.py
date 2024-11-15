import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# Load the data
data_week_1 = pd.read_csv('./Data/tracking_week_1.csv')
data_week_2 = pd.read_csv('./Data/tracking_week_2.csv')
data_week_3 = pd.read_csv('./Data/tracking_week_3.csv')
data_week_4 = pd.read_csv('./Data/tracking_week_4.csv')
data_week_5 = pd.read_csv('./Data/tracking_week_5.csv')
data_week_6 = pd.read_csv('./Data/tracking_week_6.csv')

data = pd.concat([data_week_1, data_week_2, data_week_3, data_week_4, data_week_5, data_week_6], ignore_index=True)

# Filter for the primary wide receiver using their nflId
player_ids = [43700]

# Filter data for these players
players_data = data[data['nflId'].isin(player_ids)]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
# Stratify based on players
for train_index, test_index in sss.split(data, data['nflId']):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

'''
# Set up the field dimensions for plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 120)  # Field length in yards
ax.set_ylim(0, 53.3)  # Field width in yards

# Plot each play's movement as a line, overlaying all plays
for play_id in player_data['playId'].unique():
    play_data = player_data[player_data['playId'] == play_id]
    ax.plot(play_data['x'], play_data['y'], alpha=0.9, linewidth=3)

# Customize the plot
ax.set_title('Movement Patterns of the Primary Wide Receiver Across Plays')
ax.set_xlabel('Field Length (yards)')
ax.set_ylabel('Field Width (yards)')
plt.gca().invert_yaxis()

plt.show()
'''