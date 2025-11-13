from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import io

file_name = "flashscore_stats.csv"

try:
    df = pd.read_csv(file_name)

    # eliminarea coloanelor irelevante sau lipsa
    cols_del = [
        'link_meci',
        'data_meci',
        'faulturi_gazda',
        'faulturi_oaspete'
    ]
    cols_del_ex = [col for col in cols_del if col in df.columns]
    df = df.drop(columns = cols_del_ex)

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
    input_cols = [col for col in numeric_col if col not in ['goluri_gazda', 'goluri_oaspete', 'posesie_gazda', 'posesie_oaspete']]

    for col in input_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    
    # crearea de feature-uri noi
    conditions = [
        (df['goluri_gazda'] > df['goluri_oaspete']), # Cazul 1: victorie gazda
        (df['goluri_gazda'] == df['goluri_oaspete']), # Cazul 0: Egal
        (df['goluri_gazda'] < df['goluri_oaspete']) # cazul 2: victorie oaspete
    ]

    choices = [1, 0, 2]
    df['rezultat'] = np.select(conditions, choices)

    # feature eng
    df['diff_posesie'] = df['posesie_minge_gazda'] - df['posesie_minge_oaspete']
    df['diff_total_suturi'] = df['total_suturi_gazda'] - df['total_suturi_oaspete']
    df['diff_suturi_pe_poarta'] = df['suturi_pe_poarta_gazda'] - df['suturi_pe_poarta_oaspete']
    df['diff_cornere'] = df['cornere_gazda'] - df['cornere_oaspete']
    df['diff_cartonase_galbene'] = df['cartonase_galbene_gazda'] - df['cartonase_galbene_oaspete']
    df['diff_cartonase_rosii'] = df['cartonase_rosii_gazda'] - df['cartonase_rosii_oaspete']
    df['diff_ofsaiduri'] = df['ofsaiduri_gazda'] - df['ofsaiduri_oaspete']
    df['diff_lovituri_libere'] = df['lovituri_libere_gazda'] - df['lovituri_libere_oaspete']
    df['diff_aruncari_de_la_margine'] = df['aruncari_de_la_margine_gazda'] - df['aruncari_de_la_margine_oaspete']
    df['diff_interventii_portar'] = df['interventii_portar_gazda'] - df['interventii_portar_oaspete']
    
except FileNotFoundError:
    print(f"ERROR: The file {file_name} was not found")
except Exception as e:
    print(f"ERROR: An error occured: {e}")
