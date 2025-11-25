from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import zipfile
from typing import Optional

class DataProcessor:
    def __init__(self, input_steps, output_steps, target_column_name='Global_active_power',
                 local_raw_path: Optional[str] = None,
                 local_raw_df: Optional[pd.DataFrame] = None,
                 get_all_label_features: bool = False):
        
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.target_column_name = target_column_name
        self.target_column_index = 0
        self.get_all_label_features = get_all_label_features
        self.local_raw_path = local_raw_path
        self.local_raw_df = local_raw_df

        self.scaler = StandardScaler() 

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

    def _fetch_clean_and_engineer(self):
        print("Step 1/5: Fetching, cleaning, and engineering features...")
        df = None
        if getattr(self, 'local_raw_df', None) is not None:
            df = self.local_raw_df.copy()

        if df is None and getattr(self, 'local_raw_path', None):
            lp = self.local_raw_path
            if os.path.exists(lp):
                try:
                    if lp.lower().endswith('.zip'):
                        with zipfile.ZipFile(lp, 'r') as z:
                            candidates = [n for n in z.namelist() if n.lower().endswith(('.txt', '.csv'))]
                            if not candidates: raise RuntimeError(f'No .txt/.csv files found inside zip: {lp}')
                            with z.open(candidates[0]) as fh:
                                df = pd.read_csv(fh, sep=';', header=0, decimal='.', na_values='?', low_memory=False)
                    else:
                        df = pd.read_csv(lp, sep=';', header=0, decimal='.', na_values='?', low_memory=False)
                except Exception as e:
                    print(f'Failed to read local path: {e}')
                    df = None

        if df is None:
            try:
                individual_household_electric_power_consumption = fetch_ucirepo(id=235)
                X = individual_household_electric_power_consumption.data.features
                df = X.copy()
            except Exception as e:
                raise RuntimeError(f'Failed to fetch remote dataset: {e}')
        
        # Date Time Parsing
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df = df.set_index('datetime')
            df = df.drop(['Date', 'Time'], axis=1)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.set_index('datetime')
        
        df = df.replace('?', np.nan).astype(float).ffill()
        
        # Feature Engineering
        df['Global_active_power_Wh'] = df['Global_active_power'] * 1000 / 60
        df['Sub_metering_rest'] = df['Global_active_power_Wh'] - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
        df['Sub_metering_rest'] = df['Sub_metering_rest'].clip(lower=0)
        df = df.drop('Global_active_power_Wh', axis=1)
        
        return df

    def _resample_and_reorder(self, df):
        print(f"Step 2/5: Resampling data to hourly...")
        
        agg_dict = {
            'Global_active_power': 'mean', 
            'Global_reactive_power': 'mean',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum', 
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum',
            'Sub_metering_rest': 'sum'
        }
        
        df_hourly = df.resample('H').agg(agg_dict).fillna(method='ffill')

        df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
        df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
        
        try:
            self.target_column_index = df_hourly.columns.get_loc(self.target_column_name)
        except KeyError:
            return None
        
        # Reorder so target is at index 0
        cols = [self.target_column_name] + [col for col in df_hourly.columns if col != self.target_column_name]
        df_hourly = df_hourly[cols]
        self.target_column_index = 0 
        
        return df_hourly

    def _split_and_scale(self, df):
        print("Step 3/5: Splitting and Scaling (StandardScaler)...")
        train_df = df.loc['2006-12-16':'2009-11-30']
        val_df = df.loc['2009-12-01':'2010-04-30']
        test_df = df.loc['2010-05-01':]
        
        self.scaler.fit(train_df)
        
        return self.scaler.transform(train_df), self.scaler.transform(val_df), self.scaler.transform(test_df)

    def _create_windows(self, data):
        X, y = [], []
        for i in range(len(data) - self.input_steps - self.output_steps + 1):
            input_window = data[i : (i + self.input_steps)]
            X.append(input_window)
            if self.get_all_label_features: 
                output_window = data[(i + self.input_steps) : (i + self.input_steps + self.output_steps), :]
            else:
                output_window = data[(i + self.input_steps) : (i + self.input_steps + self.output_steps), self.target_column_index]
            y.append(output_window)
        return np.array(X), np.array(y)

    def load_and_process_data(self):
        df_clean = self._fetch_clean_and_engineer()
        df_hourly = self._resample_and_reorder(df_clean)
        scaled_train, scaled_val, scaled_test = self._split_and_scale(df_hourly)
        
        print("Step 4/5: Creating windows...")
        self.X_train, self.y_train = self._create_windows(scaled_train)
        self.X_val, self.y_val = self._create_windows(scaled_val)
        self.X_test, self.y_test = self._create_windows(scaled_test)
        print("Step 5/5: Done.")
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)

    def inverse_transform_predictions(self, predictions):
        
        target_mean = self.scaler.mean_[self.target_column_index]
        target_scale = self.scaler.scale_[self.target_column_index]
        
        unscaled_predictions = (predictions * target_scale) + target_mean
        return unscaled_predictions