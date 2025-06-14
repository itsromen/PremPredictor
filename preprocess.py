import pandas as pd
import os

def preprocess(infile='Premier_League.csv', outfile='premier_league_cleaned.csv'):
    # Load raw data
    df = pd.read_csv(infile)

    # Remove rows with any NA values in core columns
    core_cols = ['Goals Home', 'Away Goals', 'attendance',
                 'home_possessions', 'away_possessions', 'home_shots', 'away_shots']
    df = df.dropna(subset=core_cols)

    # Clean attendance: remove commas and convert to int
    df['attendance'] = df['attendance'].str.replace(',', '')
    df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')

    # Ensure numeric types for stats
    numeric_cols = ['Goals Home', 'Away Goals', 'home_possessions', 'away_possessions', 'home_shots', 'away_shots']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows again if after coercion there are NaNs
    df = df.dropna(subset=numeric_cols + ['attendance'])

    # Keep original structure - don't add home_win column

    # Save cleaned CSV
    df.to_csv(outfile, index=False)
    return df

if __name__ == '__main__':
    cleaned = preprocess()
    print('Saved cleaned data with shape', cleaned.shape)
