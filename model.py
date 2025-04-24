# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os


def train_model():
    """Train and return the salary prediction model"""
    try:
        # Check if the dataset exists
        if os.path.exists('player_stats_and_salary_fixed (1).csv'):
            # Load dataset
            data = pd.read_csv('player_stats_and_salary_fixed (1).csv', low_memory=False)
            print(f"Successfully loaded data with {len(data)} rows")
        else:
            raise FileNotFoundError("Could not find player_stats_and_salary_fixed (1).csv")

        # Clean data - handle missing values
        for col in data.columns:
            if data[col].dtype != 'object':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # Fill missing numeric values with mean or 0
                data[col].fillna(data[col].mean() if data[col].mean() > 0 else 0, inplace=True)

        # Define team salary data
        team_salary_data = {
            "Phoenix": 220708856,
            "Minnesota": 204780898,
            "Boston": 195610488,
            "New York": 193588886,
            "LA Lakers": 192057940,
            "Milwaukee": 185971982,
            "Denver": 185864258,
            "Dallas": 178812859,
            "Golden State": 178316619,
            "Miami": 176102077,
            "New Orleans": 175581168,
            "LA Clippers": 174124752,
            "Philadelphia": 174059777,
            "Washington": 173873325,
            "Toronto": 173621417,
            "Sacramento": 172815356,
            "Cleveland": 172471107,
            "Charlotte": 171952448,
            "Brooklyn": 171804859,
            "Atlanta": 170056977,
            "Houston": 170038023,
            "Indiana": 169846170,
            "Portland": 169031747,
            "Chicago": 168147899,
            "Oklahoma City": 167471133,
            "Memphis": 165903638,
            "San Antonio": 164872330,
            "Utah": 156874018,
            "Orlando": 151728562,
            "Detroit": 140746162,
        }

        # Add TeamSalaryCommitment column based on team abbreviation
        team_abbr_mapping = {
            'PHO': 'Phoenix', 'PHX': 'Phoenix', 'MIN': 'Minnesota', 'BOS': 'Boston',
            'NYK': 'New York', 'LAL': 'LA Lakers', 'MIL': 'Milwaukee', 'DEN': 'Denver',
            'DAL': 'Dallas', 'GSW': 'Golden State', 'MIA': 'Miami', 'NOP': 'New Orleans',
            'LAC': 'LA Clippers', 'PHI': 'Philadelphia', 'WAS': 'Washington', 'TOR': 'Toronto',
            'SAC': 'Sacramento', 'CLE': 'Cleveland', 'CHA': 'Charlotte', 'BRK': 'Brooklyn',
            'BKN': 'Brooklyn', 'ATL': 'Atlanta', 'HOU': 'Houston', 'IND': 'Indiana',
            'POR': 'Portland', 'CHI': 'Chicago', 'OKC': 'Oklahoma City', 'MEM': 'Memphis',
            'SAS': 'San Antonio', 'UTA': 'Utah', 'ORL': 'Orlando', 'DET': 'Detroit'
        }

        if 'tm' in data.columns:
            data['TeamName'] = data['tm'].map(team_abbr_mapping)
            data['TeamSalaryCommitment'] = data['TeamName'].map(team_salary_data)
            data['TeamSalaryCommitment'].fillna(160000000, inplace=True)  # Default value
        else:
            data['TeamSalaryCommitment'] = 160000000  # Default salary commitment

        # Create additional features if needed
        if 'ts_percent' in data.columns:
            data['TS_Percentage'] = data['ts_percent']
        else:
            data['TS_Percentage'] = 0.55  # League average

        if 'per' in data.columns:
            data['Simple_PER'] = data['per']
        else:
            data['Simple_PER'] = 15.0  # League average

        # Ensure all required columns exist
        required_columns = ['points', 'assists', 'reboundsTotal', 'TS_Percentage', 'Simple_PER', 'TeamSalaryCommitment']
        for col in required_columns:
            if col not in data.columns:
                print(f"Warning: {col} not found in dataset, creating placeholder")
                data[col] = 0

        # Make sure we have a Salary column
        if 'Salary' not in data.columns:
            raise ValueError("No Salary column found in the dataset")

        # Select features and target
        features = ['points', 'assists', 'reboundsTotal', 'TS_Percentage', 'Simple_PER', 'TeamSalaryCommitment']
        target = 'Salary'

        # Train-test split
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error: ${mae:,.2f}")

        # Return the trained model
        return model

    except Exception as e:
        print(f"Error in train_model: {e}")
        # Return a simple model if training fails
        model = RandomForestRegressor(random_state=42)
        dummy_X = np.array([[20, 5, 5, 0.55, 15, 160000000]])
        dummy_y = np.array([5000000])
        model.fit(dummy_X, dummy_y)
        return model


def prepare_player_features(player_stats, player_info):
    """
    Prepare player features for salary prediction
    Parameters:
    - player_stats: DataFrame with player game log statistics
    - player_info: DataFrame with player information
    Returns:
    - DataFrame with features formatted for the model
    """
    # Team salary data for the 2024/25 season
    team_salary_data = {
        "Phoenix": 220708856,
        "Minnesota": 204780898,
        "Boston": 195610488,
        "New York": 193588886,
        "LA Lakers": 192057940,
        "Milwaukee": 185971982,
        "Denver": 185864258,
        "Dallas": 178812859,
        "Golden State": 178316619,
        "Miami": 176102077,
        "New Orleans": 175581168,
        "LA Clippers": 174124752,
        "Philadelphia": 174059777,
        "Washington": 173873325,
        "Toronto": 173621417,
        "Sacramento": 172815356,
        "Cleveland": 172471107,
        "Charlotte": 171952448,
        "Brooklyn": 171804859,
        "Atlanta": 170056977,
        "Houston": 170038023,
        "Indiana": 169846170,
        "Portland": 169031747,
        "Chicago": 168147899,
        "Oklahoma City": 167471133,
        "Memphis": 165903638,
        "San Antonio": 164872330,
        "Utah": 156874018,
        "Orlando": 151728562,
        "Detroit": 140746162,
    }

    # Get the player's team name
    team_name = player_info['TEAM_NAME'].values[0]

    # Fetch the team salary commitment, default to 120M if not found
    team_salary_commitment = team_salary_data.get(team_name, 120000000)

    # Prepare features
    features = {
        'points': player_stats['PTS'].astype(float).mean(),
        'assists': player_stats['AST'].astype(float).mean(),
        'reboundsTotal': player_stats['REB'].astype(float).mean(),
        'TS_Percentage': player_stats['PTS'].sum() / (2 * (player_stats['FGA'].sum() + 0.44 * player_stats['FTA'].sum())),
        'Simple_PER': (
            player_stats['PTS'].astype(float).mean() +
            player_stats['REB'].astype(float).mean() * 1.2 +
            player_stats['AST'].astype(float).mean() * 1.5 +
            player_stats['STL'].astype(float).mean() * 2 +
            player_stats['BLK'].astype(float).mean() * 2 -
            player_stats['TOV'].astype(float).mean() * 1.2
        ) / player_stats['MIN'].astype(float).mean(),
        'TeamSalaryCommitment': team_salary_commitment,
    }

    # Return as DataFrame
    return pd.DataFrame([features])


# If run directly, train the model
if __name__ == "__main__":
    model = train_model()
    print("Model training complete")
