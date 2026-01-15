import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

data_path = Path(__file__).parent.parent / 'data' / 'flashscore_stats.csv'
df = pd.read_csv(data_path)

df['rezultat'] = df.apply(lambda row:
    1 if row['goluri_gazda'] > row['goluri_oaspete']
    else (0 if row['goluri_gazda'] == row['goluri_oaspete'] else 2), axis=1)

rezultat_counts = df['rezultat'].value_counts().sort_index()
total_meciuri = len(df)

home_wins = rezultat_counts.get(1, 0)
draws = rezultat_counts.get(0, 0)
away_wins = rezultat_counts.get(2, 0)

home_wins_pct = (home_wins / total_meciuri) * 100
draws_pct = (draws / total_meciuri) * 100
away_wins_pct = (away_wins / total_meciuri) * 100


home_matches = df['echipa_gazda'].value_counts()
away_matches = df['echipa_oaspete'].value_counts()
total_matches_per_team = home_matches.add(away_matches, fill_value=0).sort_values(ascending=False)

echipa_cele_mai_multe = total_matches_per_team.index[0]
nr_meciuri_max = int(total_matches_per_team.iloc[0])

echipa_cele_mai_putine = total_matches_per_team.index[-1]
nr_meciuri_min = int(total_matches_per_team.iloc[-1])


home_win_matches = df[df['rezultat'] == 1]
avg_shots_winner_home = home_win_matches['suturi_pe_poarta_gazda'].mean()
avg_shots_loser_away = home_win_matches['suturi_pe_poarta_oaspete'].mean()
avg_fouls_winner_home = home_win_matches['faulturi_gazda'].mean()
avg_fouls_loser_away = home_win_matches['faulturi_oaspete'].mean()

away_win_matches = df[df['rezultat'] == 2]
avg_shots_winner_away = away_win_matches['suturi_pe_poarta_oaspete'].mean()
avg_shots_loser_home = away_win_matches['suturi_pe_poarta_gazda'].mean()
avg_fouls_winner_away = away_win_matches['faulturi_oaspete'].mean()
avg_fouls_loser_home = away_win_matches['faulturi_gazda'].mean()



avg_shots_castigatoare = (avg_shots_winner_home * len(home_win_matches) +
                          avg_shots_winner_away * len(away_win_matches)) / (len(home_win_matches) + len(away_win_matches))
avg_shots_pierzatoare = (avg_shots_loser_away * len(home_win_matches) +
                         avg_shots_loser_home * len(away_win_matches)) / (len(home_win_matches) + len(away_win_matches))
avg_fouls_castigatoare = (avg_fouls_winner_home * len(home_win_matches) +
                          avg_fouls_winner_away * len(away_win_matches)) / (len(home_win_matches) + len(away_win_matches))
avg_fouls_pierzatoare = (avg_fouls_loser_away * len(home_win_matches) +
                         avg_fouls_loser_home * len(away_win_matches)) / (len(home_win_matches) + len(away_win_matches))



df_possession = df.copy()
if df_possession['posesie_minge_gazda'].dtype == object:
    df_possession['posesie_minge_gazda'] = df_possession['posesie_minge_gazda'].str.replace('%', '').replace('', np.nan)
    df_possession['posesie_minge_gazda'] = pd.to_numeric(df_possession['posesie_minge_gazda'], errors='coerce')
if df_possession['posesie_minge_oaspete'].dtype == object:
    df_possession['posesie_minge_oaspete'] = df_possession['posesie_minge_oaspete'].str.replace('%', '').replace('', np.nan)
    df_possession['posesie_minge_oaspete'] = pd.to_numeric(df_possession['posesie_minge_oaspete'], errors='coerce')

df_possession = df_possession[df_possession['posesie_minge_gazda'].notna() &
                              df_possession['posesie_minge_oaspete'].notna()]

if len(df_possession) > 0:
    home_win_poss = df_possession[df_possession['rezultat'] == 1]
    avg_poss_winner_home = home_win_poss['posesie_minge_gazda'].mean()
    avg_poss_loser_away = home_win_poss['posesie_minge_oaspete'].mean()
    away_win_poss = df_possession[df_possession['rezultat'] == 2]
    avg_poss_winner_away = away_win_poss['posesie_minge_oaspete'].mean()
    avg_poss_loser_home = away_win_poss['posesie_minge_gazda'].mean()
    avg_poss_castigatoare = (avg_poss_winner_home * len(home_win_poss) +
                            avg_poss_winner_away * len(away_win_poss)) / (len(home_win_poss) + len(away_win_poss))
    avg_poss_pierzatoare = (avg_poss_loser_away * len(home_win_poss) +
                           avg_poss_loser_home * len(away_win_poss)) / (len(home_win_poss) + len(away_win_poss))

victorii_acasa = home_wins
victorii_deplasare = away_wins
total_victorii = victorii_acasa + victorii_deplasare

pct_victorii_acasa = (victorii_acasa / total_victorii) * 100 if total_victorii > 0 else 0
pct_victorii_deplasare = (victorii_deplasare / total_victorii) * 100 if total_victorii > 0 else 0


fig = plt.figure(figsize=(16, 12))

sizes = [home_wins, draws, away_wins]
colors = ['#2E8B57', '#FFA500', '#DC143C']
ax2 = plt.subplot(2, 3, 1)
rezultate_labels = ['Victorii\nGazdă', 'Egalități', 'Victorii\nOaspeți']
rezultate_values = [home_wins_pct, draws_pct, away_wins_pct]
bars = ax2.bar(rezultate_labels, rezultate_values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Procentaje Rezultate', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel('Procent (%)', fontweight='bold')
ax2.set_ylim(0, max(rezultate_values) * 1.2)
for i, (bar, val) in enumerate(zip(bars, rezultate_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}%\n({sizes[i]} meciuri)',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

ax3 = plt.subplot(2, 3, 2)
all_teams = total_matches_per_team
ax3.barh(range(len(all_teams)), all_teams.values, color='#4169E1', edgecolor='black')
ax3.set_yticks(range(len(all_teams)))
ax3.set_yticklabels(all_teams.index, fontsize=7)
ax3.set_xlabel('Număr Meciuri', fontweight='bold')
ax3.set_title(f'Toate Echipele după Număr de Meciuri ({len(all_teams)} echipe)', fontsize=12, fontweight='bold', pad=10)
ax3.invert_yaxis()
for i, v in enumerate(all_teams.values):
    ax3.text(v + 0.5, i, str(int(v)), va='center', fontweight='bold', fontsize=6)

ax4 = plt.subplot(2, 3, 4)
categories = ['Echipa\nCâștigătoare', 'Echipa\nPierzătoare']
shots_values = [avg_shots_castigatoare, avg_shots_pierzatoare]
bars4 = ax4.bar(categories, shots_values, color=['#228B22', '#B22222'],
                edgecolor='black', linewidth=1.5, width=0.6)
ax4.set_title('Suturi pe Poartă: Câștigător vs Învins', fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel('Medie Suturi pe Poartă', fontweight='bold')
ax4.set_ylim(0, max(shots_values) * 1.3)
for bar, val in zip(bars4, shots_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val:.2f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4 = plt.subplot(2, 3, 5)
categories = ['Echipa\nCâștigătoare', 'Echipa\nPierzătoare']
shots_values = [avg_fouls_castigatoare, avg_fouls_pierzatoare]
bars4 = ax4.bar(categories, shots_values, color=['#228B22', '#B22222'],
                edgecolor='black', linewidth=1.5, width=0.6)
ax4.set_title('Faulturi: Câștigător vs Învins', fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel('Medie Faulturi', fontweight='bold')
ax4.set_ylim(0, max(shots_values) * 1.3)
for bar, val in zip(bars4, shots_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val:.2f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax5 = plt.subplot(2, 3, 6)
if len(df_possession) > 0:
    poss_categories = ['Echipa\nCâștigătoare', 'Echipa\nPierzătoare']
    poss_values = [avg_poss_castigatoare, avg_poss_pierzatoare]
    bars5 = ax5.bar(poss_categories, poss_values, color=['#228B22', '#B22222'],
                    edgecolor='black', linewidth=1.5, width=0.6)
    ax5.set_title('Posesie Minge: Câștigător vs Învins', fontsize=12, fontweight='bold', pad=10)
    ax5.set_ylabel('Medie Posesie (%)', fontweight='bold')
    ax5.set_ylim(0, 100)
    for bar, val in zip(bars5, poss_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{val:.1f}%',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax5.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
else:
    ax5.text(0.5, 0.5, 'Nu există date\nde posesie',
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')

ax6 = plt.subplot(2, 3, 3)
victorii_labels = ['Victorii\nAcasă', 'Victorii\nDeplasare']
victorii_values = [victorii_acasa, victorii_deplasare]
bars6 = ax6.bar(victorii_labels, victorii_values, color=['#2E8B57', '#DC143C'],
                edgecolor='black', linewidth=1.5, width=0.6)
ax6.set_title('Victorii Acasă vs Victorii în Deplasare', fontsize=12, fontweight='bold', pad=10)
ax6.set_ylabel('Număr Victorii', fontweight='bold')
ax6.set_ylim(0, max(victorii_values) * 1.2)
for bar, val in zip(bars6, victorii_values):
    height = bar.get_height()
    pct = (val / total_victorii * 100) if total_victorii > 0 else 0
    ax6.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{val}\n({pct:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
analysis_folder = Path(__file__).parent / 'analysis'
analysis_folder.mkdir(exist_ok=True)
output_path = analysis_folder / 'analiza_statistici.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')