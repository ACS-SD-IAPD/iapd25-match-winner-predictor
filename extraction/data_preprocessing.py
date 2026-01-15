import pandas as pd
import numpy as np

file_name = "D:\\UPB\\IAPD\\iapd25-match-winner-predictor\\flashscore_stats.csv"

try:
    df = pd.read_csv(file_name)

    df['data_meci'] = pd.to_datetime(df['data_meci'], format='%d.%m.%Y', errors='coerce')
    df = df.sort_values('data_meci', kind='mergesort').reset_index(drop=True)
    df['season'] = df['data_meci'].dt.year + (df['data_meci'].dt.month < 8).astype(int)
    cols_pos = [
        'posesie_minge_gazda',
        'posesie_minge_oaspete'
    ]
    for col in cols_pos:
        df[col] = df[col].astype(str).str.replace('%', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    gazda_isnull = df['posesie_minge_gazda'].isnull()
    oaspete_isnull = df['posesie_minge_oaspete'].isnull()
    ambele_isnull = gazda_isnull & oaspete_isnull
    df.loc[ambele_isnull, 'posesie_minge_gazda'] = 50
    df.loc[ambele_isnull, 'posesie_minge_oaspete'] = 50
    doar_gazda_isnull = gazda_isnull & ~oaspete_isnull
    df.loc[doar_gazda_isnull, 'posesie_minge_gazda'] = 100 - df.loc[doar_gazda_isnull, 'posesie_minge_oaspete']
    doar_oaspete_isnull = ~gazda_isnull & oaspete_isnull
    df.loc[doar_oaspete_isnull, 'posesie_minge_oaspete'] = 100 - df.loc[doar_oaspete_isnull, 'posesie_minge_gazda']
    numeric_col = df.select_dtypes(include=[np.number]).columns
    input_cols = [col for col in numeric_col if col not in ['goluri_gazda', 'goluri_oaspete', 'posesie_minge_gazda', 'posesie_minge_oaspete']]

    for col in input_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    result_df = df[['link_meci', 'data_meci', 'echipa_gazda', 'echipa_oaspete', 'goluri_gazda', 'goluri_oaspete']].copy()
    result_df['meciuri_jucate_gazda_in_sezon'] = 0
    result_df['meciuri_jucate_oaspete_in_sezon'] = 0
    result_df['etapa_sezon'] = 0
    result_df['forma_puncte_gazda'] = 0
    result_df['forma_puncte_oaspete'] = 0
    result_df['h2h_victorii_gazda'] = 0

    def _points_for_team_in_match(team: str, match_row: pd.Series) -> int:
        hg = match_row['goluri_gazda']
        ag = match_row['goluri_oaspete']
        if match_row['echipa_gazda'] == team:
            if hg > ag:
                return 3
            if hg == ag:
                return 1
            return 0
        if ag > hg:
            return 3
        if ag == hg:
            return 1
        return 0

    metrics_pairs = [
        ('goluri_gazda', 'goluri_oaspete', 'medie_goluri'),
        ('goluri_oaspete', 'goluri_gazda', 'medie_goluri_primite'),

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

    for idx, row in result_df.iterrows():
        home_team = row['echipa_gazda']
        away_team = row['echipa_oaspete']
        match_date = row['data_meci']
        current_season = int(df.loc[idx, 'season'])
        home_prev_home = df[
            (df['season'] == current_season) &
            (df['echipa_gazda'] == home_team) &
            (df['data_meci'] < match_date)
        ]
        away_prev_away = df[
            (df['season'] == current_season) &
            (df['echipa_oaspete'] == away_team) &
            (df['data_meci'] < match_date)
        ]
        if len(home_prev_home) > 0:
            last_5_home = home_prev_home.tail(5)
            result_df.loc[idx, 'forma_puncte_gazda'] = int(
                sum(_points_for_team_in_match(home_team, m) for _, m in last_5_home.iterrows())
            )
        else:
            result_df.loc[idx, 'forma_puncte_gazda'] = 0

        if len(away_prev_away) > 0:
            last_5_away = away_prev_away.tail(5)
            result_df.loc[idx, 'forma_puncte_oaspete'] = int(
                sum(_points_for_team_in_match(away_team, m) for _, m in last_5_away.iterrows())
            )
        else:
            result_df.loc[idx, 'forma_puncte_oaspete'] = 0
        h2h_same_venue = df[
            (df['echipa_gazda'] == home_team) &
            (df['echipa_oaspete'] == away_team) &
            (df['data_meci'] < match_date)
        ].tail(5)

        result_df.loc[idx, 'h2h_gazda'] = int(
            ((h2h_same_venue['goluri_gazda'] > h2h_same_venue['goluri_oaspete']).sum()) - ((h2h_same_venue['goluri_gazda'] < h2h_same_venue['goluri_oaspete']).sum()) > 0
        )

        for gazda_col, oaspete_col, metric_name in metrics_pairs:
            if len(home_prev_home) > 0:
                last_5_home = home_prev_home.tail(5)
                result_df.loc[idx, metric_name + '_gazda'] = np.mean(last_5_home[gazda_col].astype(float)) if len(last_5_home) > 0 else 0
            else:
                result_df.loc[idx, metric_name + '_gazda'] = 0
            if len(away_prev_away) > 0:
                last_5_away = away_prev_away.tail(5)
                result_df.loc[idx, metric_name + '_oaspete'] = np.mean(last_5_away[oaspete_col].astype(float)) if len(last_5_away) > 0 else 0
            else:
                result_df.loc[idx, metric_name + '_oaspete'] = 0

    for col in result_df.columns:
        if col not in ['link_meci', 'data_meci', 'echipa_gazda', 'echipa_oaspete', 'goluri_gazda', 'goluri_oaspete']:
            result_df[col] = result_df[col].round(1)

    conditions = [
        (result_df['goluri_gazda'] > result_df['goluri_oaspete']),
        (result_df['goluri_gazda'] == result_df['goluri_oaspete']),
        (result_df['goluri_gazda'] < result_df['goluri_oaspete'])
    ]
    choices = [1, 0, 2]
    result_df['rezultat'] = np.select(conditions, choices)

    result_df['season'] = result_df['data_meci'].dt.year + (result_df['data_meci'].dt.month < 8).astype(int)

    for idx, row in result_df.iterrows():
        current_season = row['season']
        match_date = row['data_meci']
        home_team = row['echipa_gazda']
        away_team = row['echipa_oaspete']

        prev_season_matches = result_df[(result_df['season'] == current_season) & (result_df['data_meci'] < match_date)]

        home_played = int(((prev_season_matches['echipa_gazda'] == home_team) | (prev_season_matches['echipa_oaspete'] == home_team)).sum())
        away_played = int(((prev_season_matches['echipa_gazda'] == away_team) | (prev_season_matches['echipa_oaspete'] == away_team)).sum())

        result_df.loc[idx, 'meciuri_jucate_gazda_in_sezon'] = home_played
        result_df.loc[idx, 'meciuri_jucate_oaspete_in_sezon'] = away_played

        result_df.loc[idx, 'etapa_sezon'] = max(home_played, away_played) + 1

    result_df['loc_clasament_gazda'] = 0
    result_df['loc_clasament_oaspete'] = 0

    for idx, row in result_df.iterrows():
        home_team = row['echipa_gazda']
        away_team = row['echipa_oaspete']
        current_season = row['season']
        match_date = row['data_meci']

        season_matches = result_df[(result_df['season'] == current_season) & (result_df['data_meci'] < match_date)]
        standings = {}
        season_teams = set(result_df[result_df['season'] == current_season]['echipa_gazda'].unique()) | \
                       set(result_df[result_df['season'] == current_season]['echipa_oaspete'].unique())
        
        for team in season_teams:
            standings[team] = {'points': 0, 'matches': 0, 'goals_for': 0, 'goals_against': 0}
        for _, match in season_matches.iterrows():
            h_team = match['echipa_gazda']
            a_team = match['echipa_oaspete']
            result = match['rezultat']
            goals_h = match['goluri_gazda']
            goals_a = match['goluri_oaspete']
            
            if result == 1:
                standings[h_team]['points'] += 3
            elif result == 0:
                standings[h_team]['points'] += 1
                standings[a_team]['points'] += 1
            else:
                standings[a_team]['points'] += 3
            
            standings[h_team]['matches'] += 1
            standings[a_team]['matches'] += 1
            standings[h_team]['goals_for'] += goals_h
            standings[h_team]['goals_against'] += goals_a
            standings[a_team]['goals_for'] += goals_a
            standings[a_team]['goals_against'] += goals_h

        sorted_standings = sorted(standings.items(), key=lambda x: (-x[1]['points'], -(x[1]['goals_for'] - x[1]['goals_against'])))
        standings_list = [team for team, _ in sorted_standings]
        
        result_df.loc[idx, 'loc_clasament_gazda'] = standings_list.index(home_team) + 1 if home_team in standings_list else 0
        result_df.loc[idx, 'loc_clasament_oaspete'] = standings_list.index(away_team) + 1 if away_team in standings_list else 0
    result_df = result_df.drop(columns=['goluri_gazda', 'goluri_oaspete', 'season'])
    result_df = result_df[result_df['etapa_sezon'] > 5].copy()

    result_df = result_df.drop(columns=['etapa_sezon', 'meciuri_jucate_gazda_in_sezon', 'meciuri_jucate_oaspete_in_sezon'])
    result_df.to_csv('team_form.csv', index=False)
    print("Successfully created team_form.csv with team form statistics")

except FileNotFoundError:
    print(f"ERROR: The file {file_name} was not found")
except Exception as e:
    print(f"ERROR: An error occured: {e}")
