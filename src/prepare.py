import pandas as pd
import os

def prepare_data(input_path):
    df = pd.read_csv(input_path)
    
    # column names
    column_mapping = {
        'Ticker': 'ticker',
        'TickerType': 'ticker_type',
        'Company': 'company',
        'Traded': 'traded',
        'Transaction': 'transaction',
        'Trade_Size_USD': 'trade_size_usd',
        'Status': 'status',
        'Subholding': 'subholding',
        'Description': 'description',
        'Name': 'name',
        'BioGuideID': 'bioguideid',
        'Filed': 'filed',
        'Party': 'party',
        'District': 'district',
        'Chamber': 'chamber',
        'Comments': 'comments',
        'Quiver_Upload_Time': 'quiver_upload_time',
        'excess_return': 'excess_return',
        'State': 'state',
        'last_modified': 'last_modified'
    }
    df = df.rename(columns=column_mapping)
    
    # dates
    df['disclosure_date'] = pd.to_datetime(df['disclosure_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # create lag
    df['reporting_lag'] = (df['disclosure_date'] - df['transaction_date']).dt.days
    
    # split dataset
    # dev: 2020-2024 | locked test: 2025-2026
    dev_set = df[df['disclosure_date'] < '2025-01-01']
    test_set = df[df['disclosure_date'] >= '2025-01-01']
    
    # save files
    os.makedirs('data/processed', exist_ok=True)
    dev_set.to_csv('data/processed/dev_set.csv', index=False)
    test_set.to_csv('data/processed/locked_test_set.csv', index=False)
    
    print(f"Prepared {len(dev_set)} training rows and {len(test_set)} locked test rows.")

if __name__ == "__main__":
    prepare_data('data/raw/your_manual_export.csv')
