from xml.etree.ElementInclude import include
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np
import io

file_name = "flashscore_stats.csv"

try:
    df = pd.read_csv(file_name)

    # Convert date to datetime and sort chronologically
    df['data_meci'] = pd.to_datetime(df['data_meci'], format='%d.%m.%Y', errors='coerce')
    df = df.sort_values('data_meci').reset_index(drop=True)

    # convertirea text->numeric si eliminarea simbolului '%'
    cols_pos = [
        'posesie_minge_gazda',
        'posesie_minge_oaspete'
    ]
    for col in cols_pos:
        # elim '%'
        df[col] = df[col].astype(str).str.replace('%', '')

        # conv in numeric
        df[col] = pd.to_numeric(df[col], errors='coerce') # textul care nu e numar va fi trcut ca NaN

    # daca un rand din posesie contine NaN, tratam 3 cazuri
    gazda_isnull = df['posesie_minge_gazda'].isnull()
    oaspete_isnull = df['posesie_minge_oaspete'].isnull()

    #1: ambele sunt NaN
    ambele_isnull = gazda_isnull & oaspete_isnull
    df.loc[ambele_isnull, 'posesie_minge_gazda'] = 50
    df.loc[ambele_isnull, 'posesie_minge_oaspete'] = 50

    #2: doar gazda e NaN
    doar_gazda_isnull = gazda_isnull & ~oaspete_isnull
    df.loc[doar_gazda_isnull, 'posesie_minge_gazda'] = 100 - df.loc[doar_gazda_isnull, 'posesie_minge_oaspete']

    #3: doar oaspete e NaN
    doar_oaspete_isnull = ~gazda_isnull & oaspete_isnull
    df.loc[doar_oaspete_isnull, 'posesie_minge_oaspete'] = 100 - df.loc[doar_oaspete_isnull, 'posesie_minge_gazda']

    # valorile NaN din coloanele ramase vor fi umplute cu mediana
    numeric_col = df.select_dtypes(include=[np.number]).columns
    input_cols = [col for col in numeric_col if col not in ['goluri_gazda', 'goluri_oaspete', 'posesie_minge_gazda', 'posesie_minge_oaspete']]

    for col in input_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Create dataframe with match info and team form
    result_df = df[['link_meci', 'data_meci', 'echipa_gazda', 'echipa_oaspete', 'goluri_gazda', 'goluri_oaspete']].copy()

    # Calculate rolling averages for each team (form up to each match date)
    # Metrics to calculate averages for
    metrics_pairs = [
        ('suturi_pe_poarta_gazda', 'suturi_pe_poarta_oaspete', 'medie_suturi_pe_poarta'),
        ('total_suturi_gazda', 'total_suturi_oaspete', 'medie_total_suturi'),
        ('cornere_gazda', 'cornere_oaspete', 'medie_cornere'),
        ('cartonase_galbene_gazda', 'cartonase_galbene_oaspete', 'medie_cartonase_galbene'),
        ('cartonase_rosii_gazda', 'cartonase_rosii_oaspete', 'medie_cartonase_rosii'),
        ('ofsaiduri_gazda', 'ofsaiduri_oaspete', 'medie_ofsaiduri'),
        ('lovituri_libere_gazda', 'lovituri_libere_oaspete', 'medie_lovituri_libere'),
        ('interventii_portar_gazda', 'interventii_portar_oaspete', 'medie_interventii_portar'),
        ('posesie_minge_gazda', 'posesie_minge_oaspete', 'medie_posesie_minge')
    ]

    # For each row, calculate form stats for home and away team
    for idx, row in result_df.iterrows():
        home_team = row['echipa_gazda']
        away_team = row['echipa_oaspete']
        match_date = row['data_meci']
        
        # Get all previous matches for home team (where they were home OR away)
        home_prev = df[((df['echipa_gazda'] == home_team) | (df['echipa_oaspete'] == home_team)) & (df['data_meci'] < match_date)]
        away_prev = df[((df['echipa_gazda'] == away_team) | (df['echipa_oaspete'] == away_team)) & (df['data_meci'] < match_date)]
        
        # Calculate averages for each metric
        for gazda_col, oaspete_col, metric_name in metrics_pairs:
            # For home team (last 5 matches)
            if len(home_prev) > 0:
                last_5_home = home_prev.tail(5)
                # Calculate average possession for this team across all their matches
                values = []
                for _, match in last_5_home.iterrows():
                    if match['echipa_gazda'] == home_team:
                        values.append(match[gazda_col])
                    else:  # home_team played as oaspete
                        values.append(match[oaspete_col])
                result_df.loc[idx, metric_name + '_gazda'] = np.mean(values) if len(values) > 0 else 0
            else:
                result_df.loc[idx, metric_name + '_gazda'] = 0
            
            # For away team (last 5 matches)
            if len(away_prev) > 0:
                last_5_away = away_prev.tail(5)
                # Calculate average for away_team across all their matches
                values = []
                for _, match in last_5_away.iterrows():
                    if match['echipa_gazda'] == away_team:
                        values.append(match[gazda_col])
                    else:  # away_team played as oaspete
                        values.append(match[oaspete_col])
                result_df.loc[idx, metric_name + '_oaspete'] = np.mean(values) if len(values) > 0 else 0
            else:
                result_df.loc[idx, metric_name + '_oaspete'] = 0

    # Round values to 1 decimal place
    for col in result_df.columns:
        if col not in ['link_meci', 'data_meci', 'echipa_gazda', 'echipa_oaspete', 'goluri_gazda', 'goluri_oaspete']:
            result_df[col] = result_df[col].round(1)
    
    # Add result column (1 = home win, 0 = draw, 2 = away win)
    conditions = [
        (result_df['goluri_gazda'] > result_df['goluri_oaspete']),
        (result_df['goluri_gazda'] == result_df['goluri_oaspete']),
        (result_df['goluri_gazda'] < result_df['goluri_oaspete'])
    ]
    choices = [1, 0, 2]
    result_df['rezultat'] = np.select(conditions, choices)
    
    # Add season column - determine based on month (seasons typically start in August)
    result_df['season'] = result_df['data_meci'].dt.year + (result_df['data_meci'].dt.month < 8).astype(int)
    
    # Calculate standings before each match
    # Add standings columns
    result_df['loc_clasament_gazda'] = 0
    result_df['loc_clasament_oaspete'] = 0
    
    # For each match, calculate standings before it (only for same season)
    for idx, row in result_df.iterrows():
        home_team = row['echipa_gazda']
        away_team = row['echipa_oaspete']
        current_season = row['season']
        match_date = row['data_meci']
        
        # Get all matches from same season before this match
        season_matches = result_df[(result_df['season'] == current_season) & (result_df['data_meci'] < match_date)]
        
        # Build standings only from season matches
        standings = {}
        
        # Get unique teams in this season
        season_teams = set(result_df[result_df['season'] == current_season]['echipa_gazda'].unique()) | \
                       set(result_df[result_df['season'] == current_season]['echipa_oaspete'].unique())
        
        for team in season_teams:
            standings[team] = {'points': 0, 'matches': 0, 'goals_for': 0, 'goals_against': 0}
        
        # Calculate standings from all previous matches in this season
        for _, match in season_matches.iterrows():
            h_team = match['echipa_gazda']
            a_team = match['echipa_oaspete']
            result = match['rezultat']
            goals_h = match['goluri_gazda']
            goals_a = match['goluri_oaspete']
            
            if result == 1:  # Home win
                standings[h_team]['points'] += 3
            elif result == 0:  # Draw
                standings[h_team]['points'] += 1
                standings[a_team]['points'] += 1
            else:  # Away win
                standings[a_team]['points'] += 3
            
            standings[h_team]['matches'] += 1
            standings[a_team]['matches'] += 1
            standings[h_team]['goals_for'] += goals_h
            standings[h_team]['goals_against'] += goals_a
            standings[a_team]['goals_for'] += goals_a
            standings[a_team]['goals_against'] += goals_h
        
        # Sort standings
        sorted_standings = sorted(standings.items(), key=lambda x: (-x[1]['points'], -(x[1]['goals_for'] - x[1]['goals_against'])))
        standings_list = [team for team, _ in sorted_standings]
        
        result_df.loc[idx, 'loc_clasament_gazda'] = standings_list.index(home_team) + 1 if home_team in standings_list else 0
        result_df.loc[idx, 'loc_clasament_oaspete'] = standings_list.index(away_team) + 1 if away_team in standings_list else 0
    
    # Remove goal columns and season column before saving
    result_df = result_df.drop(columns=['goluri_gazda', 'goluri_oaspete', 'season'])
    
    # Save to CSV
    result_df.to_csv('team_form.csv', index=False)
    print("Successfully created team_form.csv with team form statistics")
    
except FileNotFoundError:
    print(f"ERROR: The file {file_name} was not found")
except Exception as e:
    print(f"ERROR: An error occured: {e}")
