# NBA Salary Predictor Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
import requests
import json
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo, playergamelog
import plotly.express as px
import os
from google import genai

# Import directly from model.py
import model4 as model

# Set page configuration
st.set_page_config(page_title="NBA Salary Predictor and Performance Analysis", page_icon="🏀", layout="wide")

# App title and description
st.title("NBA Player Salary Predictor")
st.markdown("Predict NBA player salaries based on performance statistics")

# Load the model
@st.cache_resource
def load_model():
    try:
        # Import the train_model function directly from model.py
        model_instance = model.train_model2()
        return model_instance
    except Exception as e:
        st.error(f"Error loading/training model: {e}")
        st.code(traceback.format_exc())
        return None

# Fetch player stats using the NBA API - optimized for recent seasons
def get_player_stats(player_id):
    """
    Fetch player stats from NBA API.
    Only returns data if player has played in recent seasons (2019-20 onwards).
    For players who have played recently, returns ALL of their historical data.
    
    Parameters:
    - player_id: The NBA player ID
    
    Returns:
    - DataFrame with player game statistics, or empty DataFrame if player hasn't played recently
    """
    # First, check if the player has played in recent seasons (2019-20 onwards)
    recent_seasons = [
        "2023-24",
        "2022-23",
        "2021-22", 
        "2020-21",
        "2019-20"
    ]
    
    has_recent_data = False
    
    # Check if player has any games in recent seasons
    for season in recent_seasons:
        try:
            # Quick check for recent activity
            recent_check = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            recent_df = pd.DataFrame(recent_check.get_data_frames()[0])
            
            if not recent_df.empty:
                has_recent_data = True
                break
                
        except Exception as e:
            print(f"Error checking {season} season data for player {player_id}: {e}")
    
    # If player hasn't played recently, return empty DataFrame
    if not has_recent_data:
        print(f"Player {player_id} has no recent data (2019-20 onwards)")
        return pd.DataFrame()
    
    # For players with recent data, get ALL historical data
    try:
        # Now fetch ALL seasons since the player has recent data
        game_logs = playergamelog.PlayerGameLog(player_id=player_id, season="ALL")
        df = pd.DataFrame(game_logs.get_data_frames()[0])
        
        if not df.empty and 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Sort by game date (most recent first)
            df = df.sort_values('GAME_DATE', ascending=False)
        
        return df
        
    except Exception as e:
        print(f"Error fetching complete stats for player {player_id}: {e}")
        return pd.DataFrame()

def predict_salary_with_cap_and_tiers(player_features, model_instance, info_df, player_name=None):
    """
    Predict player salary using percentage of cap method with appropriate tiers,
    ensuring top players get the max tier
    Parameters:
    - player_features: DataFrame with player statistics
    - model_instance: Trained model for salary prediction
    - info_df: DataFrame with player information
    - player_name: (Optional) Player's name to check for All-NBA status
    """
    # List of features the model was trained with
    expected_features = ['per', 'ts_percent', 'ws', 'bpm', 'age', 'experience', 'mp']
    
    if model_instance is None:
        return None
        
    # Create a DataFrame with only the expected features
    prediction_df = pd.DataFrame()

    # Add only the features the model expects
    for feature in expected_features:
        if feature in player_features.columns:
            prediction_df[feature] = player_features[feature]
        else:
            # Map from player_features to expected_features or use default
            if feature == 'per':
                prediction_df[feature] = player_features['Simple_PER'] if 'Simple_PER' in player_features.columns else 15.0
            elif feature == 'ts_percent':
                prediction_df[feature] = player_features['TS_Percentage'] if 'TS_Percentage' in player_features.columns else 0.55
            elif feature == 'mp':
                prediction_df[feature] = 25.0  # Default minutes per game
            elif feature == 'age':
                prediction_df[feature] = 28.0  # Default age
            elif feature == 'experience':
                prediction_df[feature] = info_df['SEASON_EXP'].values[0] if 'SEASON_EXP' in info_df.columns else 5.0
            elif feature == 'ws':
                prediction_df[feature] = 3.0  # Default win shares
            elif feature == 'bpm':
                prediction_df[feature] = 0.0  # Default box plus/minus
            else:
                prediction_df[feature] = 0.0

  
    if prediction_df.empty:
        print("Error: The prediction DataFrame is empty.")
        return None
    
    # Calculate player_rating safely
    try:
        player_rating = 0.0
        
        # Use PER (Player Efficiency Rating)
        if 'per' in prediction_df.columns:
            per_value = prediction_df['per'].values[0]
            player_rating += per_value * 2.5  # PER of 20 contributes 50 points
            
        # Use TS% (True Shooting Percentage)
        if 'ts_percent' in prediction_df.columns:
            ts_value = prediction_df['ts_percent'].values[0]
            player_rating += ts_value * 50  # TS% of 0.60 contributes 30 points
            
        # Use Win Shares
        if 'ws' in prediction_df.columns:
            ws_value = prediction_df['ws'].values[0]
            player_rating += ws_value * 5  # WS of 6 contributes 30 points
            
        # Use Box Plus/Minus
        if 'bpm' in prediction_df.columns:
            bpm_value = prediction_df['bpm'].values[0]
            player_rating += bpm_value * 2.5  # BPM of 6 contributes 15 points
            
        # Add minutes played factor
        if 'mp' in prediction_df.columns:
            mp_value = prediction_df['mp'].values[0]
            player_rating += min(mp_value/36 * 10, 10)  # Up to 10 points for playing time
    # Normalize the rating to ensure it's reasonable
        player_rating = max(min(player_rating, 100), 0)  # Clamp between 0 and 100
        print(f"Calculated player_rating: {player_rating}")
    except Exception as e:
        print(f"Error calculating player rating: {e}")
        player_rating = 25.0  # Default rating if calculation fails
    
    # Current NBA salary cap
    NBA_SALARY_CAP = 140588000  # 2024-25 season

    # NBA salary constants for tiers
    MIN_SALARY_PCT = 2.5  # Tier 1: Minimum salary (approximate percentage)
    BAE_PCT = 3.32  # Tier 2: Bi-annual exception
    ROOM_PCT = 5.678  # Tier 3: Room exception
    MLE_PCT = 9.0  # Tier 4: Mid-Level Exception

    # Check if player name was provided
    player_is_all_nba = False
    if player_name:
        # All-NBA Players (2023-24 and 2022-23 seasons)
        all_nba_players = [
            # 2023-24 First Team
            "Giannis Antetokounmpo", "Luka Dončić", "Shai Gilgeous-Alexander",
            "Nikola Jokić", "Jayson Tatum",
            # 2023-24 Second Team
            "Jalen Brunson", "Anthony Davis", "Kevin Durant",
            "Anthony Edwards", "Kawhi Leonard",
            # 2023-24 Third Team
            "Devin Booker", "Stephen Curry", "Tyrese Haliburton",
            "LeBron James", "Domantas Sabonis",
            # 2022-23 First Team
            "Joel Embiid",
            # 2022-23 Second Team
            "Jimmy Butler", "Jaylen Brown", "Donovan Mitchell",
            # 2022-23 Third Team
            "Julius Randle", "De'Aaron Fox", "Damian Lillard"
        ]

        # Standardize player name for comparison
        for i, p in enumerate(all_nba_players):
            # Remove diacritics for more reliable comparison
            all_nba_players[i] = p.replace("č", "c").replace("ć", "c")

        # Check if player is in All-NBA list (with standardized name)
        standardized_player_name = player_name.replace("č", "c").replace("ć", "c")
        player_is_all_nba = any(name.lower() in standardized_player_name.lower() for name in all_nba_players)

    # REMOVE THIS DUPLICATE DEFINITION
    # List of features the model was trained with
    # expected_features = ['per', 'ts_percent', 'ws', 'bpm', 'age', 'experience', 'mp']

    # Get the player's experience
    experience = int(info_df['SEASON_EXP'].values[0])

    # Determine max eligible percentage based on experience
    if experience <= 6:
        max_eligible_pct = 25.0  # 0-6 years: 25% of cap
    elif experience <= 9:
        max_eligible_pct = 30.0  # 7-9 years: 30% of cap
    else:
        max_eligible_pct = 35.0  # 10+ years: 35% of cap
    
    # Thresholds for top players (adjust these values based on your data)
    TOP_PLAYER_THRESHOLD = 50.0  # Players above this rating get max contract
    HIGH_TIER_THRESHOLD = 40.0  # Players above this get 25-30% tier
     # Make initial prediction
    initial_predicted_salary = model_instance.predict(prediction_df)[0]
    # Convert to percentage of salary cap
    predicted_pct = (initial_predicted_salary / NBA_SALARY_CAP) * 100
    
    # If player is All-NBA, automatically assign max contract
    if player_is_all_nba:
        salary_pct = max_eligible_pct
        tier = 9  # Max tier
        tier_name = f"Maximum Contract ({max_eligible_pct}% of Cap) - All-NBA Player"
        # Set a high player rating for consistency
        player_rating = 100.0
    # Determine the salary tier and adjusted percentage based on rating and prediction
    elif player_rating >= TOP_PLAYER_THRESHOLD:
        # Top players always get max eligible percentage
        salary_pct = ((max_eligible_pct * .8)+(predicted_pct * .2))
        tier = 9  # Max tier
        tier_name = f"Maximum Contract ({max_eligible_pct}% of Cap)"
    elif player_rating >= HIGH_TIER_THRESHOLD:
        # High tier players get 25-30% (capped at their max eligible)
        salary_pct = ((max_eligible_pct * .7) + (predicted_pct * .3))
        tier = 8  # 25-30% tier
    elif predicted_pct >= 20.0:
        salary_pct = ((max_eligible_pct * .5) + (predicted_pct * .5))  # Middle of Tier 7
        tier = 7  # 20-25% tier
    elif predicted_pct >= 15.0:
        salary_pct = ((max_eligible_pct * .4) + (predicted_pct * .6))  # Middle of Tier 6
        tier = 6  # 15-20% tier
        tier_name = "15-20% of Cap"
    elif predicted_pct >= 10.0:
        salary_pct = ((max_eligible_pct * .3) + (predicted_pct * .7))  # Middle of Tier 5
        tier = 5  # 10-15% tier
        tier_name = "10-15% of Cap"
    elif abs(predicted_pct - MLE_PCT) <= 1.5:  # Within 1.5% of MLE
        salary_pct = MLE_PCT
        tier = 4  # MLE tier
        tier_name = "Mid-Level Exception"
    elif abs(predicted_pct - ROOM_PCT) <= 1.0:  # Within 1.0% of Room Exception
        salary_pct = ROOM_PCT
        tier = 3  # Room exception tier
        tier_name = "Room Exception"
    elif abs(predicted_pct - BAE_PCT) <= 0.7:  # Within 0.7% of BAE
        salary_pct = BAE_PCT
        tier = 2  # BAE tier
        tier_name = "Bi-Annual Exception"
    elif predicted_pct <= 3.0:
        salary_pct = MIN_SALARY_PCT
        tier = 1  # Minimum salary tier
        tier_name = "Minimum Salary"
    else:
        salary_pct = predicted_pct
        tier = 0  # No specific tier
        tier_name = "Role Player"

    # Calculate final salary
    final_salary = NBA_SALARY_CAP * (salary_pct / 100)
    result = {
        'Salary': final_salary,
        'Percentage': salary_pct,
        'Tier': tier,
        'TierName': tier_name,
        'Experience': experience,
        'MaxEligible': max_eligible_pct,
        'PlayerRating': player_rating,  # Include player rating for reference
        'IsAllNBA': player_is_all_nba  # Add flag for All-NBA status
    }
    return result

# Use the prepare_player_features function from model.py
prepare_player_features = model.prepare_player_features

# Main app logic
def main():
    # Load the model
    model_instance = load_model()
    with st.expander("All-NBA Selection Information"):
        st.markdown("""
        **2023-24 All-NBA Teams:**

        **First Team:**
        - Giannis Antetokounmpo (Bucks)
        - Luka Dončić (Mavericks)
        - Shai Gilgeous-Alexander (Thunder)
        - Nikola Jokić (Nuggets)
        - Jayson Tatum (Celtics)

        **Second Team:**
        - Jalen Brunson (Knicks)
        - Anthony Davis (Lakers)
        - Kevin Durant (Suns)
        - Anthony Edwards (Timberwolves)
        - Kawhi Leonard (Clippers)

        **Third Team:**
        - Devin Booker (Suns)
        - Stephen Curry (Warriors)
        - Tyrese Haliburton (Pacers)
        - LeBron James (Lakers)
        - Domantas Sabonis (Kings)

        **2022-23 All-NBA Teams (selected players not in 2023-24):**
        - Joel Embiid (76ers)
        - Jimmy Butler (Heat)
        - Jaylen Brown (Celtics)
        - Donovan Mitchell (Cavaliers)
        - Julius Randle (Knicks)
        - De'Aaron Fox (Kings)
        - Damian Lillard (Trail Blazers)
        """)
    # Get active players first
    player_dict = players.get_players()
    active_players = [p for p in player_dict if p.get('is_active', True)]
    
    # Get player names and sort alphabetically
    player_names = [player['full_name'] for player in active_players]
    player_names.sort()
    
    # Set default player
    default_player = "Lebron James" 
    default_index = player_names.index(default_player) if default_player in player_names else 0
    
    # Player selection
    selected_player = st.selectbox("Select Player:", player_names, index=default_index)

    if selected_player:
        # Get player ID
        player_id = [p for p in active_players if p['full_name'] == selected_player][0]['id']
        
        with st.spinner("Loading player data..."):
            # Fetch player stats - this now only gets stats if player has recent data
            player_stats = get_player_stats(player_id)
            has_recent_stats = not player_stats.empty
            
            # Always get player info
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = pd.DataFrame(player_info.get_data_frames()[0])

            # Calculate age
            birth_date = pd.to_datetime(info_df['BIRTHDATE'].values[0])
            age = (pd.Timestamp.now() - birth_date).days / 365.25

        # Always show the player profile
        st.subheader("Player Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {selected_player}")
            st.write(f"**Team:** {info_df['TEAM_NAME'].values[0]}")
            st.write(f"**Position:** {info_df['POSITION'].values[0]}")
            st.write(f"**Experience:** {info_df['SEASON_EXP'].values[0]} years")
            st.write(f"**Height:** {info_df['HEIGHT'].values[0]}")
            st.write(f"**Weight:** {info_df['WEIGHT'].values[0]} lbs")
            st.write(f"**Country:** {info_df['COUNTRY'].values[0]}")
            st.write(f"**Draft Year:** {info_df['DRAFT_YEAR'].values[0]}")
            st.write(f"**Age:** {age:.1f} years")
        with col2:
            headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
            st.image(headshot_url, use_container_width=True)

        # Only proceed with analysis if player has recent stats
        if has_recent_stats:
            st.markdown("---")
            st.subheader(f"{selected_player}'s Salary and Performance Analysis")
            
            tab1, tab2, tab3 = st.tabs(["**Salary Prediction**", "**Performance Statistics**", "**Contract Information**"])
            
            with tab1:
                player_features = prepare_player_features(player_stats, info_df)
                # Pass the player name to the prediction function
                predicted_result = predict_salary_with_cap_and_tiers(player_features, model_instance, info_df, selected_player)
                if predicted_result is not None:
                    # Check if player is All-NBA
                    is_all_nba = predicted_result.get('IsAllNBA', False)

                    # Display the predicted salary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("**Predicted Salary (2024-2025 Season)**", f"${predicted_result['Salary']:,.2f}")
                    with col2:
                        st.metric("**Percentage of Salary Cap**", f"{predicted_result['Percentage']:.2f}%")

                    # Display All-NBA badge if applicable
                    if is_all_nba:
                        st.markdown(
                            """
                            <div style="background-color: #FFD700; color: #000000; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center;">
                            <h3 style="margin: 0;">🏆 ALL-NBA SELECTION (2022-24) 🏆</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Display tier information with appropriate styling
                    tier_colors = {
                        1: "#E0E0E0",  # Minimum
                        2: "#C8E6C9",  # BAE
                        3: "#B3E5FC",  # Room
                        4: "#FFECB3",  # MLE
                        5: "#D1C4E9",  # 10-15%
                        6: "#BBDEFB",  # 15-20%
                        7: "#FFCCBC",  # 20-25%
                        8: "#F8BBD0",  # 25-30%
                        9: "#FFD180",  # Max
                        0: "#E0E0E0",  # Custom/Other
                    }
                    tier_color = tier_colors.get(predicted_result['Tier'], "#E0E0E0")
                    st.markdown(
                        f"""
                        <div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                        <h3 style="margin: 0;">Salary Tier: {predicted_result['TierName']}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # For max contract players, show additional context
                    if predicted_result['Tier'] == 9:
                        if is_all_nba:
                            st.info(f"Based on All-NBA selection and {predicted_result['Experience']} years of experience, " +
                                    f"{selected_player} is eligible for a maximum of {predicted_result['MaxEligible']}% of the salary cap.")
                        else:
                            st.info(f"Based on {predicted_result['Experience']} years of experience, " +
                                    f"{selected_player} is eligible for a maximum of {predicted_result['MaxEligible']}% of the salary cap.")

                    # Add context about the salary cap
                    st.caption(f"The NBA Salary Cap for the 2024-25 season is ${140588000:,.2f}")

                    # Display key factors
                    st.subheader("Key Factors Influencing Salary")
                    # Create a bar chart of key stats
                    key_stats = ['points', 'assists', 'reboundsTotal', 'Simple_PER', 'TS_Percentage']
                    available_stats = [stat for stat in key_stats if stat in player_features.columns]
                    if available_stats:
                        # Prepare data for the bar chart
                        bar_data = pd.DataFrame({
                            'Stat': available_stats,
                            'Value': [player_features[stat].values[0] for stat in available_stats]
                        })

                        # Map the original stat names to more descriptive names
                        stat_name_mapping = {
                            'points': 'Points Per Game (PPG)',
                            'assists': 'Assists Per Game (APG)',
                            'reboundsTotal': 'Rebounds Per Game (RPG)',
                            'Simple_PER': 'Player Efficiency Rating (PER)',
                            'TS_Percentage': 'True Shooting Percentage (TS%)'
                        }

                        # Apply the mapping to the Stat column
                        bar_data['Stat'] = bar_data['Stat'].map(stat_name_mapping)

                        # Create the bar chart
                        fig = px.bar(
                            bar_data,
                            x='Stat',
                            y='Value',
                            title="Key Factors Influencing Salary",
                            text='Value'
                        )

                        # Update the text formatting and position
                        fig.update_traces(
                            texttemplate='%{text:.2f}',  # Format text to 2 decimal places
                            textposition='outside'       # Position text outside the bars
                        )

                        # Update layout for better readability
                        fig.update_layout(
                            yaxis_title="Metric Value",          # Set y-axis title
                            xaxis_title="Performance Metrics",   # Set x-axis title
                            title_font_size=18,                  # Increase title font size
                            xaxis_tickangle=-45,                 # Rotate x-axis labels for better readability
                            showlegend=False                     # Hide legend
                        )

                        # Display the chart
                        st.plotly_chart(fig)

                    # Add an explanation box for the stats
                    with st.expander("Explanation of Key Stats"):
                        st.markdown("""
                        - **Points (PPG):** Average points scored per game.
                        - **Assists (APG):** Average assists made per game.
                        - **Rebounds (RPG):** Average total rebounds (offensive + defensive) per game.
                        - **True Shooting Percentage (TS%):** A measure of shooting efficiency that accounts for field goals, three-pointers, and free throws.
                        - **Simple PER:** A simplified version of the Player Efficiency Rating, which evaluates a player's overall performance per minute.
                        - **Team Salary Commitment:** The total salary cap committed by the player's team.
                        """)

                    # Add explanation of salary tiers
                    with st.expander("Explanation of Salary Tiers"):
                        st.markdown("""
                        Salary tiers used in this prediction model:
                        | Tier | Description | % of Cap | 2024-25 Value |
                        |------|-------------|----------|---------------|
                        | 1 | Minimum Salary | ~2.5% | ~$3.5 million |
                        | 2 | Bi-Annual Exception | 3.32% | $4.7 million |
                        | 3 | Room Exception | 5.678% | $8.0 million |
                        | 4 | Mid-Level Exception | 9% | $12.8 million |
                        | 5 | 10-15% of Cap | 10-15% | $14.0-21.0 million |
                        | 6 | 15-20% of Cap | 15-20% | $21.0-28.0 million |
                        | 7 | 20-25% of Cap | 20-25% | $28.0-35.0 million |
                        | 8 | 25-30% of Cap | 25-30% | $35.0-42.0 million |
                        | 9 | Maximum Contract | 30-35% | $42.0-49.2 million |

                        **Maximum Contract Eligibility:**
                        - 0-6 years experience: 25% of cap max
                        - 7-9 years experience: 30% of cap max
                        - 10+ years experience: 35% of cap max

                        **All-NBA Players:**
                        Players who were selected to an All-NBA team (First, Second, or Third) in the 2022-23 or 2023-24 seasons automatically qualify for a maximum contract based on their years of experience.
                        """)

            with tab2:
                st.subheader(f"{selected_player}'s Game Statistics")
                min_date = player_stats['GAME_DATE'].min().date()
                max_date = player_stats['GAME_DATE'].max().date()

                # Define the default range for the 2024-2025 season
                default_start_date = pd.to_datetime("2024-10-01").date()  # Start of the 2024-2025 season
                default_end_date = pd.to_datetime("2025-04-30").date()

                # Ensure the default range is within the available data range
                default_start_date = max(default_start_date, min_date)
                default_end_date = min(default_end_date, max_date)

                start_date, end_date = st.slider("Select Date Range:", min_date, max_date, value=(default_start_date, default_end_date))
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)

                filtered_stats = player_stats[(player_stats['GAME_DATE'] >= pd.to_datetime(start_date)) & (player_stats['GAME_DATE'] <= pd.to_datetime(end_date))]

                # Season Averages
                st.subheader("Season Averages")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PPG", f"{filtered_stats['PTS'].astype(float).mean():.1f}")
                with col2:
                    st.metric("RPG", f"{filtered_stats['REB'].astype(float).mean():.1f}")
                with col3:
                    st.metric("APG", f"{filtered_stats['AST'].astype(float).mean():.1f}")
                with col4:
                    st.metric("MPG", f"{filtered_stats['MIN'].astype(float).mean():.1f}")

                # Performance Charts
                # Performance Charts
                col1, col2 = st.columns(2)

                with col1:
                    # Scoring Trend
                    fig1 = px.line(
                        filtered_stats,
                        x='GAME_DATE',
                        y='PTS',
                        title='Scoring Trend Over Time',
                        labels={'GAME_DATE': 'Game Date', 'PTS': 'Points Scored'}
                    )
                    st.plotly_chart(fig1)
                    st.caption("This chart shows the player's scoring trend over the selected date range.")

                    # Overall Performance
                    filtered_stats['PERFORMANCE'] = (
                        filtered_stats['PTS'].astype(float) +
                        filtered_stats['REB'].astype(float) +
                        filtered_stats['AST'].astype(float)
                    )
                    fig2 = px.scatter(
                        filtered_stats,
                        x='MIN',
                        y='PERFORMANCE',
                        title='Minutes Played vs Overall Performance',
                        labels={'MIN': 'Minutes Played', 'PERFORMANCE': 'Overall Performance'}
                    )
                    st.plotly_chart(fig2)
                    st.caption("This chart highlights the relationship between minutes played and overall performance.")

                with col2:
                    # Minutes Played
                    fig3 = px.line(
                        filtered_stats,
                        x='GAME_DATE',
                        y='MIN',
                        title='Minutes Played Per Game',
                        labels={'GAME_DATE': 'Game Date', 'MIN': 'Minutes Played'}
                    )
                    st.plotly_chart(fig3)
                    st.caption("This chart shows the player's minutes played per game over the selected date range.")

                    # Efficiency
                    filtered_stats['EFFICIENCY'] = filtered_stats['PERFORMANCE'] / filtered_stats['MIN'].astype(float)
                    fig4 = px.line(
                        filtered_stats,
                        x='GAME_DATE',
                        y='EFFICIENCY',
                        title='Player Efficiency Over Time',
                        labels={'GAME_DATE': 'Game Date', 'EFFICIENCY': 'Efficiency'}
                    )
                    st.plotly_chart(fig4)
                    st.caption("This chart tracks the player's efficiency over the selected date range.")

            with tab3:
                with st.spinner("Analyzing contract information..."):
                    try:
                        # Initialize the Google GenAI client
                        #api_key = st.secrets['client']['api_key']
                        #client = genai.Client(api_key=api_key)
                        client = genai.Client(api_key="AIzaSyAz56zLp5egYUz_2jGNTDYMddJW9KXNu88")
                        # Create a more structured prompt to get consistent JSON
                        prompt = f"""
                        Provide a detailed analysis of {selected_player}'s current NBA contract.
                        Return ONLY a valid JSON object with this exact structure:
                        {{
                        "current_contract": {{
                        "years": "Number of years",
                        "total_value": "Total contract value",
                        "annual_average": "Average annual value",
                        "team": "Current team"
                        }},
                        "contract_history": [
                        {{
                        "period": "Years with team",
                        "value": "Contract value",
                        "team": "Team name"
                        }}
                        ],
                        "bonuses": "Brief description of known salary bonuses",
                        "comparison": "Brief comparison to similar players in the NBA",
                        "salary_cap": "Current salary cap",
                        "cap_impact": "Brief description of salary cap impact on player and team"
                        }}
                        Do not include any explanations, markdown formatting, or code blocks in your response - ONLY the JSON object.
                        """

                        # Generate response using the model
                        response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt)
                        
                        # Get the response text
                        response_text = response.text.strip()
                        
                        # Clean up the response text to ensure valid JSON
                        if response_text.startswith('```json'):
                            response_text = response_text.replace('```json', '', 1)
                        if response_text.endswith('```'):
                            response_text = response_text.rsplit('```', 1)[0]
                        
                        # Remove any backticks
                        response_text = response_text.strip('`').strip()

                        try:
                            # Parse the JSON
                            contract_data = json.loads(response_text)

                            # Display contract information
                            st.subheader("Current Contract")
                            current = contract_data.get("current_contract", {})
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Term:** {current.get('years', 'N/A')}")
                                st.write(f"**Total Value:** {current.get('total_value', 'N/A')}")
                            with col2:
                                st.write(f"**Annual Average:** {current.get('annual_average', 'N/A')}")
                                st.write(f"**Team:** {current.get('team', 'N/A')}")

                            # Display contract history
                            st.subheader("Contract History")
                            history = contract_data.get("contract_history", [])
                            if isinstance(history, list) and history:
                                for contract in history:
                                    st.write(f"**{contract.get('period', 'N/A')}:** {contract.get('value', 'N/A')} ({contract.get('team', 'N/A')})")
                            else:
                                st.write("No contract history available.")

                            # Display bonuses
                            st.subheader("Performance Bonuses")
                            bonuses = contract_data.get("bonuses", "No bonus information available.")
                            st.write(bonuses)

                            # Display comparison
                            st.subheader("Market Comparison")
                            comparison = contract_data.get("comparison", "No comparison information available.")
                            st.write(comparison)

                            # Display cap impact
                            st.subheader("Salary Cap Impact")
                            cap_impact = contract_data.get("cap_impact", "No salary cap impact information available.")
                            st.write(cap_impact)

                        
                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse JSON response: {e}")
                            st.write("Raw response (for debugging):")
                            st.code(response_text)

                            # Try to display the information in a more readable format
                            st.subheader("Contract Information (Unformatted)")
                            st.write(response_text)
                    except Exception as e:
                        st.error(f"Error generating contract analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                st.markdown("---")
                st.subheader("Strategic Analysis")
                
                with st.spinner("Generating player analysis..."):
                    try:
                        # Create a focused prompt for player analysis
                        prompt = f"""
                        Provide a concise analysis of {selected_player} covering:
                        1. Contract situation and market value
                        2. Performance strengths and weaknesses
                        3. Team fit and strategic recommendations
                        
                        Keep the analysis factual, data-driven, and easy to understand.
                        """
                        
                        # Generate analysis
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt
                        )
                        
                        analysis = response.text.strip()
                        
                        # Display analysis in a clean, organized format
                        st.markdown(analysis)
                        
                        # Add useful reference links
                        st.markdown("---")
                        st.markdown("### Reference Links")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("[NBA.com](https://www.nba.com)")
                            st.markdown("[Basketball Reference](https://www.basketball-reference.com/players/)")
                        with col2:
                            st.markdown("[NBA Salary Cap Info](https://www.nba.com/news/free-agency-explained)")
                            st.markdown("[Player News & Updates](https://www.nba.com/players)")
                        
                    except Exception as e:
                        st.error("Could not generate player analysis")
                        with st.expander("View technical details"):
                            st.code(traceback.format_exc())
                            # Add a disclaimer
                    st.markdown(
                        """
                        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.8em; margin-top: 20px;">
                        <strong>Note:</strong> This contract analysis was generated using AI and may not reflect the most current information. Please verify details with official NBA sources.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
if __name__ == "__main__":
    main()
