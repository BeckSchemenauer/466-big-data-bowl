import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data

# Define bucket ranges
def categorize_yardline(yardline, ):
    if yardline > 100:
        return ">100 yards to go"
    elif 80 <= yardline <= 100:
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
        print(f"{yardline} yards to go ")
        return 'Out of range'


def player_tracking_data(tracking_week_x):
    tracking_data = tracking_week_x[tracking_week_x['nflId'] == player_id]
    return tracking_data

def standardize_data(tracking_data):
    # Standardize the player's direction to always appear as moving to the right
    tracking_data['standardized_x'] = tracking_data.apply(
        lambda row: 120 - row['x'] if row['playDirection'] == 'left' else row['x'], axis=1
    )
    return tracking_data

def graph_tracking_data(tracking_data, player_id, player_position, standardized_x=False, bucket_name=None):
    # Set up the field dimensions for plotting
    start = time.time()


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 120)  # Field length in yards
    ax.set_ylim(0, 53.3)  # Field width in yards

    # Add endzone lines
    ax.axvline(10, color='red', linestyle='--', linewidth=2, label='Endzone Boundary')
    ax.axvline(110, color='red', linestyle='--', linewidth=2)

    # Add endzone labels
    ax.text(5, 26.65, "Endzone", color='red', fontsize=12, ha='center', rotation=90, verticalalignment='center')
    ax.text(115, 26.65, "Endzone", color='red', fontsize=12, ha='center', rotation=90, verticalalignment='center')

    # Plot each play's movement as a line, overlaying all plays
    plays = tracking_data['playId'].unique()
    play_count = len(plays)

    if standardized_x:
        x = 'standardized_x'
    else:
        x = 'x'

    for play_id in plays:
        play_data = tracking_data[tracking_data['playId'] == play_id]
        player_play_data = player_play[(player_play['nflId'] == player_id) & (player_play['playId'] == play_id)]
        if player_position != 'QB':
            reception = True if player_play_data.iloc[0]["hadPassReception"] == 1 else False
        else:
            reception = True

        # Generate a unique color for this play
        color = plt.cm.viridis(np.random.rand())
        if reception:
            catch_data = tracking_data[(tracking_data['playId'] == play_id) &
                                       (tracking_data['event'] == 'pass_outcome_caught')]
            try:
                catch_x = catch_data[x].iloc[0]
            except IndexError:
                print(f"reception on Play: {play_id}")
                catch_x = None
            try:
                catch_y = catch_data['y'].iloc[0]
            except IndexError:
                catch_y = None
            color = (*color[:3], 0.8)
        else:
            color = (*color[:3], 0.1)  # Adjust alpha to 0.5

        # Plot the trajectory line using the unique color
        ax.plot(play_data[x], play_data['y'], linewidth=3, color=color)
        if reception:
            ax.scatter(catch_x, catch_y, color='black', marker='x', s=50, label='Catch', zorder=3)
        # Add a dot for the starting point with the same color as the line
        ax.scatter(play_data[x].iloc[0], play_data['y'].iloc[0], color=color,
                   label='Start' if play_id == plays[0] else "")

        # Add an arrow for the ending point with the same color as the line
        ax.arrow(
            play_data[x].iloc[-2], play_data['y'].iloc[-2],  # Start of arrow (next-to-last point)
            play_data[x].iloc[-1] - play_data[x].iloc[-2],  # X-displacement
            play_data['y'].iloc[-1] - play_data['y'].iloc[-2],  # Y-displacement
            head_width=1, head_length=2, fc=color, ec=color,
            label='End' if play_id == plays[0] else ""
        )

    # Customize the plot
    title = f'Standardized Movement Patterns of {player_name} Across {play_count} Plays'
    if bucket_name:
        title += f' ({bucket_name})'
    ax.set_title(title)
    ax.set_xlabel('Field Length (yards)')
    ax.set_ylabel('Field Width (yards)')
    plt.gca().invert_yaxis()

    plt.show()
    end = time.time()
    print(f"Graphed data in {(end - start):.2f} seconds")


def calculate_yards_to_go(row):
    # Determine the absolute yardline based on the side of the field
    if row['yardlineSide'] == row['possessionTeam']:
        absolute_yardline = 100 - row['yardlineNumber']  # Offensive half
    else:
        absolute_yardline = row['yardlineNumber']       # Defensive half

    # Yards to go are relative to the offense's goal line
    if absolute_yardline > 100:
        print(">100 yards to go")
    elif absolute_yardline < 0:
        print("<0 yards to go")
    return absolute_yardline

def graph_by_yard_bucket(tracking_data, player_id, player_position, standardized_x=False):
    # Apply the function to calculate yards to go
    tracking_data['yards_to_go'] = tracking_data.apply(calculate_yards_to_go, axis=1)
    tracking_data['yard_bucket'] = tracking_data['yards_to_go'].apply(categorize_yardline)
    yard_buckets = tracking_data['yard_bucket'].unique()

    for bucket in yard_buckets:
        bucket_data = tracking_data[tracking_data['yard_bucket'] == bucket]
        graph_tracking_data(bucket_data, player_id, player_position, standardized_x, bucket_name=bucket)


def load_and_process_data(weeks, weeks_graphed):
    # Initialize an empty list to store processed data from each week
    all_tracking_data = []

    # Loop through each week and process
    for week in weeks[:weeks_graphed]:
        start = time.time()

        # Load tracking data for the current week
        tracking_data = pd.read_csv(f'./Data/tracking_week_{week}.csv')

        # isolate player tracking data
        tracking_data = player_tracking_data(tracking_data)

        # Standardize data (if needed)
        tracking_data = standardize_data(tracking_data)

        # Merge tracking data with play data
        tracking_data = tracking_data.merge(
            plays[['gameId', 'playId', 'yardlineNumber', 'yardlineSide', 'possessionTeam']],
            on=['gameId', 'playId'],
            how='left'
        )

        # Add week as a new column to distinguish between weeks
        tracking_data['week'] = week

        # Append the current week's data to the list
        all_tracking_data.append(tracking_data)

        # Print the loading time for the current week
        end = time.time()
        print(f"Processed {week} in {(end - start):.2f} seconds")

    # Concatenate all weeks' data into one DataFrame
    full_tracking_data = pd.concat(all_tracking_data, ignore_index=True)

    return full_tracking_data


if __name__ == "__main__":
    start = time.time()
    #tracking_week_1 = pd.read_csv('./Data/tracking_week_1.csv')
    players = pd.read_csv('./Data/players.csv')
    plays = pd.read_csv('./Data/plays.csv')
    player_play = pd.read_csv('./Data/player_play.csv')
    end = time.time()
    print(f"Loaded data in {(end - start):.2f} seconds")

    player_id = 53434
    player_data = players[players['nflId'] == player_id]
    player_dict = player_data.to_dict('records')[0]
    player_name = player_dict['displayName']
    player_position = player_dict['position']


    weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Add more weeks as needed
    weeks_to_graph = 9
    combined_tracking_data = load_and_process_data(weeks, weeks_to_graph)

    # Graph data for each yard bucket
    graph_by_yard_bucket(combined_tracking_data, player_id, player_position, standardized_x=True)


