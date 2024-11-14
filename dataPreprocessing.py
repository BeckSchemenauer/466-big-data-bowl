import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./Data/tracking_week_1.csv')

# Filter for the primary wide receiver using their nflId
player_id = 35459
player_data = data[data['nflId'] == player_id]

# Set up the field dimensions for plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 120)  # Field length in yards
ax.set_ylim(0, 53.3)  # Field width in yards

# Plot each play's movement as a line, overlaying all plays
for play_id in player_data['playId'].unique():
    play_data = player_data[player_data['playId'] == play_id]
    ax.plot(play_data['x'], play_data['y'], alpha=0.2, linewidth=3)

# Customize the plot
ax.set_title('Movement Patterns of the Primary Wide Receiver Across Plays')
ax.set_xlabel('Field Length (yards)')
ax.set_ylabel('Field Width (yards)')
plt.gca().invert_yaxis()

plt.show()
