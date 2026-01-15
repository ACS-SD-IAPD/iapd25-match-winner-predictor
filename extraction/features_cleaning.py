import pandas as pd
import numpy as np

from global_vars import BASE_DIR


def preprocess_data(df):
    # Elimină coloane non-numerice și identificatori
    df_processed = pd.DataFrame()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_processed['gazda_id'] = le.fit_transform(df['echipa_gazda'])
    df_processed['oaspete_id'] = le.fit_transform(df['echipa_oaspete'])
    # 1. Features diferențiale între echipe
    df_processed['diff_forma_puncte'] = df['forma_puncte_gazda'] - df['forma_puncte_oaspete']
    df_processed['diff_suturi_pe_poarta'] = df['medie_suturi_pe_poarta_gazda'] - df['medie_suturi_pe_poarta_oaspete']
    df_processed['diff_pozitie_clasament'] = df['loc_clasament_gazda'] - df['loc_clasament_oaspete']
    df_processed['diff_goluri_marcate'] = df['medie_goluri_gazda'] - df['medie_goluri_oaspete']
    df_processed['diff_goluri_primite'] = df['medie_goluri_primite_gazda'] - df['medie_goluri_primite_oaspete']
    # 4. Tratează missing values
    df_processed = df_processed.fillna(df_processed.median(numeric_only=True))

    df_processed['rezultat'] = df['rezultat']

    return df_processed

# Încarcă datele
df = pd.read_csv(BASE_DIR + '/preprocess/team_form.csv')

# Preprocesează
df_processed = preprocess_data(df)
df_processed.to_csv('team_form_final.csv', index=False)