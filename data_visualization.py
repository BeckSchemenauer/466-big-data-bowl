import time

import pandas as pd
import matplotlib.pyplot as plt

# Load the data

start = time.time()
data = pd.read_csv('./Data/tracking_week_1.csv')
players = pd.read_csv('./Data/players.csv')
end = time.time()
print(f"Loaded data in {(end - start):.2f} seconds")

# Define bucket ranges
def categorize_yardline(yardline):
    if 80 <= yardline <= 100:
        return '100-80 yards to go'
    elif 60 <= yardline < 80:
        return '80-60 yards to go'
    elif 40 <= yardline < 60:
        return '60-40 yards to go'
    elif 20 <= yardline < 40:
        return '40-20 yards to go'
    elif 0 <= yardline < 20:
        return '20-0 yards to go'
    else:
        return 'Out of range'

def group_by_yards():
    data['yard_bucket'] = data['absoluteYardlineNumber'].apply(categorize_yardline)
    grouped_data = data.groupby('yard_bucket')

    play_counts = grouped_data.size()
    play_counts.plot(kind='bar', title="Number of Plays by Yardline Buckets", xlabel="Yard Buckets", ylabel="Number of Plays")
    plt.show()

def player_tracking_data():
    player_data = players[players['nflId'] == player_id]
    player_dict = player_data.to_dict('records')[0]
    player_name = player_dict['displayName']

    tracking_data = data[data['nflId'] == player_id]
    return tracking_data, player_name

def standardize_data(tracking_data):
    # Standardize the player's direction to always appear as moving to the right
    start = time.time()
    tracking_data['standardized_x'] = tracking_data.apply(
        lambda row: 120 - row['x'] if row['playDirection'] == 'left' else row['x'], axis=1
    )
    end = time.time()
    print(f"Standardized x in {(end - start):.2f} seconds")
    return tracking_data


def graph_tracking_data(tracking_data):
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

if __name__ == "__main__":
    player_id = 47847

    tracking_data, player_name = player_tracking_data()
    tracking_data = standardize_data(tracking_data)

    #group_by_yards()

    graph_tracking_data(tracking_data)
