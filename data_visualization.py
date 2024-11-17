import time

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
start = time.time()
data = pd.read_csv('./Data/tracking_week_1.csv')
players = pd.read_csv('./Data/players.csv')
end = time.time()
print(f"Loaded data in {(end - start):.2f} seconds")

# Filter for the primary wide receiver using their nflId
player_id = 47847

player_data = players[players['nflId'] == player_id]
player_dict = player_data.to_dict('records')[0]
player_name = player_dict['displayName']

tracking_data = data[data['nflId'] == player_id]

# Standardize the player's direction to always appear as moving to the right
start = time.time()
tracking_data['standardized_x'] = tracking_data.apply(
    lambda row: 120 - row['x'] if row['playDirection'] == 'left' else row['x'], axis=1
)
end = time.time()
print(f"Standardized x in {(end - start):.2f} seconds")

# Set up the field dimensions for plotting
start = time.time()
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 120)  # Field length in yards
ax.set_ylim(0, 53.3)  # Field width in yards

# Plot each play's movement as a line, overlaying all plays
plays = tracking_data['playId'].unique()
play_count = len(plays)

for play_id in plays:
    play_data = tracking_data[tracking_data['playId'] == play_id]
    # Plot the trajectory line using standardized x
    ax.plot(play_data['standardized_x'], play_data['y'], alpha=0.2, linewidth=3)

    # Add a dot for the starting point
    ax.scatter(play_data['standardized_x'].iloc[0], play_data['y'].iloc[0], color='green',
               label='Start' if play_id == tracking_data['playId'].unique()[0] else "")

    # Add an arrow for the ending point
    ax.arrow(
        play_data['standardized_x'].iloc[-2], play_data['y'].iloc[-2],  # Start of arrow (next-to-last point)
        play_data['standardized_x'].iloc[-1] - play_data['standardized_x'].iloc[-2],  # X-displacement
        play_data['y'].iloc[-1] - play_data['y'].iloc[-2],  # Y-displacement
        head_width=1, head_length=2, fc='red', ec='red',
        label='End' if play_id == tracking_data['playId'].unique()[0] else ""
    )

# Customize the plot
ax.set_title(f'Standardized Movement Patterns of {player_name} Across {play_count} Plays')
ax.set_xlabel('Field Length (yards)')
ax.set_ylabel('Field Width (yards)')
plt.gca().invert_yaxis()

# Add a legend to clarify the start and end markers
#ax.legend(loc='bottom left')

plt.show()
end = time.time()
print(f"Graphed data in {(end - start):.2f} seconds")
