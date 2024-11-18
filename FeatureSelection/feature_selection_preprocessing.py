import os
import pandas as pd


def filter_by_wr():
    # Load the necessary CSV files
    tracking_df = pd.read_csv('../Data/tracking_week_1.csv')
    players_df = pd.read_csv('../Data/players.csv', usecols=['nflId', 'position'])

    # Filter players to include only Wide Receivers (WR)
    wr_players = players_df[players_df['position'] == 'WR']

    # Merge tracking data with WR players on nflId
    wr_tracking_df = pd.merge(tracking_df, wr_players, on='nflId', how='inner')

    # Save the filtered DataFrame to a new CSV file
    wr_tracking_df.to_csv('../Data/tracking_week_1_wr_only.csv', index=False)

    print("Filtered CSV with only WR position data saved as 'Data/tracking_week_1_wr_only.csv'.")


def combine_tracking_data(tracking_file, plays_file, player_play_file, players_file):
    # Read in the CSV files with only the specified columns
    tracking_df = pd.read_csv(tracking_file, usecols=[
        'gameId', 'playId', 'nflId', 'displayName', 'playDirection', 'x', 'y', 'frameId', 'frameType'
    ])

    player_play_df = pd.read_csv(player_play_file, usecols=[
        'gameId', 'playId', 'nflId', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset'
    ])

    plays_df = pd.read_csv(plays_file, usecols=[
        'gameId', 'playId', 'down', 'yardsToGo', 'absoluteYardlineNumber', 'gameClock',
        'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'pff_passCoverage', 'pff_manZone',
        'receiverAlignment', 'preSnapHomeTeamWinProbability',
        'preSnapVisitorTeamWinProbability', 'expectedPoints'
    ])

    players_df = pd.read_csv(players_file, usecols=['nflId', 'position'])

    # Filter players to include only Wide Receivers (WR)
    wr_players = players_df[players_df['position'] == 'WR']

    # Merge tracking data with WR players on nflId
    wr_tracking_df = pd.merge(tracking_df, wr_players, on='nflId', how='inner')

    # Merge wr_tracking_df and player_play on playId and nflId
    combined_df = pd.merge(wr_tracking_df, player_play_df, on=['gameId', 'playId', 'nflId'], how='left')

    # Merge the result with plays on playId
    combined_df = pd.merge(combined_df, plays_df, on=['gameId', 'playId'], how='left')

    # Write the final DataFrame to a CSV file
    combined_df.to_csv('../Data/relevant_wr_data.csv', index=False)

    print("Filtered and combined data saved as 'Data/relevant_wr_data.csv'.")
    return combined_df


def filter_after_snap():
    # Step 1: Remove entries where frameType is "BEFORE SNAP"
    df = combined_df[combined_df['frameType'] != 'BEFORE SNAP']

    # Step 2: Initialize an empty list to collect filtered rows
    filtered_rows = []

    # Step 3: Loop through unique combinations of gameId, playId, and nflId
    for (game_id, play_id, nfl_id) in df[['gameId', 'playId', 'nflId']].drop_duplicates().values:
        # Filter the DataFrame to only include rows for the current combination
        subset = df[(df['gameId'] == game_id) & (df['playId'] == play_id) & (df['nflId'] == nfl_id)]

        # Always include entries where frameType is "SNAP"
        snap_rows = subset[subset['frameType'] == 'SNAP']

        # Find the index of the first "AFTER SNAP" entry
        after_snap_start_idx = subset[subset['frameType'] == 'AFTER_SNAP'].index[0]

        # Select every tenth row starting from the first "AFTER SNAP" entry
        filtered_subset = subset.loc[after_snap_start_idx::10]

        # Combine SNAP and filtered AFTER SNAP rows for the current combination
        combined_subset = pd.concat([snap_rows, filtered_subset])

        # Append the combined rows to the list
        filtered_rows.append(combined_subset)

    # Step 4: Concatenate all the filtered subsets into a single DataFrame
    df_filtered = pd.concat(filtered_rows).reset_index(drop=True)

    # Write the filtered DataFrame to a CSV file
    df_filtered.to_csv('../AfterSnap/after_snap_all_data.csv', index=False)

    print("Filtered data to only include SNAP and every 10th AFTER_SNAP frame. Saved as 'filtered_data.csv'.")


def save_after_snap():
    df_filtered = pd.read_csv('../AfterSnap/after_snap_all_data.csv')

    # Step 1: Initialize a dictionary to store each new DataFrame
    snap_after_snap_dfs = {}

    # Step 2: Loop through unique combinations of gameId, playId, and nflId
    for (game_id, play_id, nfl_id) in df_filtered[['gameId', 'playId', 'nflId']].drop_duplicates().values:
        # Filter the DataFrame to only include rows for the current combination
        subset = df_filtered[
            (df_filtered['gameId'] == game_id) & (df_filtered['playId'] == play_id) & (df_filtered['nflId'] == nfl_id)]

        # Get the SNAP row(s)
        snap_rows = subset[subset['frameType'] == 'SNAP']

        # Get all AFTER SNAP rows and reset their index
        after_snap_rows = subset[subset['frameType'] == 'AFTER_SNAP'].reset_index(drop=True)

        # For each consecutive occurrence of AFTER SNAP, add to the corresponding DataFrame in the dictionary
        for i in range(len(after_snap_rows)):
            # Combine the SNAP row(s) with the i-th AFTER SNAP row
            combined_df = pd.concat([snap_rows, after_snap_rows.iloc[[i]]])

            # Append to the existing DataFrame in the dictionary if it exists, or create a new one
            if f'df_{i + 1}' in snap_after_snap_dfs:
                snap_after_snap_dfs[f'df_{i + 1}'] = pd.concat([snap_after_snap_dfs[f'df_{i + 1}'], combined_df])
            else:
                snap_after_snap_dfs[f'df_{i + 1}'] = combined_df

    # Reset index for each DataFrame in the dictionary after concatenation
    snap_after_snap_dfs = {key: df.reset_index(drop=True) for key, df in snap_after_snap_dfs.items()}

    # Loop through snap_after_snap_dfs and save each DataFrame to a CSV file
    for i, df in enumerate(snap_after_snap_dfs.values()):
        # Save each DataFrame with the specified filename format
        df.to_csv(f"../AfterSnap/after_snap_{i + 1}.csv", index=False)

    print("Split after_snap data into frame splits, saved in AfterSnap/after_snap_n.csv.")


def alter_after_snap():
    # Directory containing the CSV files
    directory = "../AfterSnap"

    # Loop through each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Load the file into a DataFrame
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)

            # Initialize lists to store the offset values
            x_offsets = []
            y_offsets = []

            # Loop through entries, calculating offsets for every second entry in pairs
            for i in range(1, len(df), 2):
                # Calculate the x and y offsets relative to the previous entry
                x_offset = df.loc[i, 'x'] - df.loc[i - 1, 'x']
                y_offset = df.loc[i, 'y'] - df.loc[i - 1, 'y']

                # If playDirection is 'left', flip the sign of x_offset
                if df.loc[i, 'playDirection'] == 'left':
                    x_offset = -x_offset

                # Store the calculated offsets
                x_offsets.append(x_offset)
                y_offsets.append(y_offset)

                # Update the DataFrame with the offsets for the second entry in each pair
                df.loc[i, 'x_offset'] = x_offset
                df.loc[i, 'y_offset'] = y_offset

            # Fill x_offset and y_offset columns with NaN for the first entries in pairs
            df['x_offset'] = df.get('x_offset', pd.Series([None] * len(df)))
            df['y_offset'] = df.get('y_offset', pd.Series([None] * len(df)))

            # Save the modified DataFrame back to a CSV file, overwriting the original file
            df.to_csv(filepath, index=False)

    print("Altered after_snap folder to normalize x_offset relative to drive direction.")


# Usage
filter_by_wr()
combined_df = combine_tracking_data('../Data/tracking_week_1_wr_only.csv', '../Data/plays.csv', '../Data/player_play.csv',
                                    '../Data/players.csv')
filter_after_snap()
save_after_snap()
alter_after_snap()
print()
