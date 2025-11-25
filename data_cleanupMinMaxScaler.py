from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import zipfile
from typing import Optional

class DataProcessor:
    """
    This class encapsulates all data preprocessing. 

    It performs:
    1. Data Loading
    2. Cleaning (NaNs, dtypes)
    3. Feature Engineering (Sub_metering_rest, Time Cycles)
    4. Resampling (to hourly)
    5. Splitting (Train, Val, Test)
    6. Scaling (MinMaxScaler)
    7. Windowing (for supervised learning)
    """
    
    def __init__(self, input_steps, output_steps, target_column_name='Global_active_power',
                 local_raw_path: Optional[str] = None,
                 local_raw_df: Optional[pd.DataFrame] = None,
                 get_all_label_features: bool = False):
        """
        Initializes the processor with windowing and target parameters.
        
        Args:
            input_steps (int): The number of past time steps to use as input (X).
            output_steps (int): The number of future time steps to predict (y).
            target_column_name (str): The name of the target variable to predict.
            get_all_label_features (bool): Whether to include all label features in the output.
        """
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.target_column_name = target_column_name
        # Will be set after resampling
        self.target_column_index = 0
        self.get_all_label_features = get_all_label_features

        # local raw data options (prefer provided DataFrame, then path)
        self.local_raw_path = local_raw_path
        self.local_raw_df = local_raw_df

        # Initialize the scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Placeholders for the final, windowed data
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

    def _fetch_clean_and_engineer(self):
        """
        Private method to perform the first block of data prep:
        Loading, timestamping, cleaning, and feature engineering.
        """
        print("Step 1/5: Fetching, cleaning, and engineering features...")
        
        # First, try using a provided local DataFrame
        df = None
        if getattr(self, 'local_raw_df', None) is not None:
            df = self.local_raw_df.copy()

        # Next, try to load from a local path if provided
        if df is None and getattr(self, 'local_raw_path', None):
            lp = self.local_raw_path
            if os.path.exists(lp):
                try:
                    # If it's a zip file, try to locate a .txt inside and read it (semicolon-separated)
                    if lp.lower().endswith('.zip'):
                        with zipfile.ZipFile(lp, 'r') as z:
                            # pick the first .txt or .csv file inside
                            candidates = [n for n in z.namelist() if n.lower().endswith(('.txt', '.csv'))]
                            if not candidates:
                                raise RuntimeError(f'No .txt/.csv files found inside zip: {lp}')
                            member = candidates[0]
                            with z.open(member) as fh:
                                df = pd.read_csv(fh, sep=';', header=0, decimal='.', na_values='?', low_memory=False)
                    else:
                        ext = os.path.splitext(lp)[1].lower()
                        if ext in ('.parquet', '.parq'):
                            df = pd.read_parquet(lp)
                        elif ext in ('.pkl', '.pickle'):
                            df = pd.read_pickle(lp)
                        else:
                            # assume csv-like
                            df = pd.read_csv(lp, sep=';', header=0, decimal='.', na_values='?', low_memory=False)
                except Exception as e:
                    print(f'Failed to read local_raw_path "{lp}": {e}. Falling back to remote fetch.')
                    df = None
            else:
                print(f'local_raw_path does not exist: {lp}. Falling back to remote fetch.')

        # Finally, fallback to remote fetch
        if df is None:
            try:
                individual_household_electric_power_consumption = fetch_ucirepo(id=235)
                X = individual_household_electric_power_consumption.data.features
                df = X.copy()
            except Exception as e:
                raise RuntimeError(
                    'Failed to fetch remote dataset via ucimlrepo.fetch_ucirepo. '
                    'If you are offline or the remote service is unavailable, supply a local file path '
                    'to the original dataset zip/txt via DataProcessor(local_raw_path=...) or an already-loaded DataFrame via local_raw_df. '
                    f'Original error: {e}'
                )
        
        # Combine Date and Time into a single datetime index if needed
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df = df.set_index('datetime')
            df = df.drop(['Date', 'Time'], axis=1)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.set_index('datetime')
        else:
            # try to convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    print('Warning: could not parse Date/Time into a datetime index. Proceeding with existing index.')
        
        # Replace '?' with a real NaN (Not a Number) value 
        df = df.replace('?', np.nan)
        
        # Convert all columns to be numeric (float)
        df = df.astype(float)
        
        # Use forward-fill to fill gaps with the last known value
        df = df.ffill()
        
        # Convert Global_active_power from kW to Watt-minutes
        df['Global_active_power_Wh'] = df['Global_active_power'] * 1000 / 60
        
        # Create the "rest of the house" (unmetered) feature
        df['Sub_metering_rest'] = df['Global_active_power_Wh'] - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
        
        # Handle any negative values that might result from measurement noise
        df['Sub_metering_rest'] = df['Sub_metering_rest'].clip(lower=0)
        
        # Drop the intermediate column, no longer needed
        df = df.drop('Global_active_power_Wh', axis=1)
        
        return df

    def _resample_and_reorder(self, df):
        """
        Private method to resample the data and place the target column first.
        """
        print(f"Step 2/5: Resampling data to hourly and setting '{self.target_column_name}' as target...")
        
        # Resample data so that we track with each hour. 
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
        
        df_hourly = df.resample('H').agg(agg_dict)
        df_hourly = df_hourly.fillna(method='ffill')

        # Calculate the hour and day from the index
        # Transform them using Sin/Cos so the model understands the cyclical nature
        # (e.g. 23:00 is close to 00:00, and Sunday is close to Monday)
        
        # 1. Hour of Day (0-23)
        df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
        
        # 2. Day of Week (0=Monday, 6=Sunday)
        df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
        df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
        # -------------------------------------
        
        # Find the index of the target column
        try:
            self.target_column_index = df_hourly.columns.get_loc(self.target_column_name)
        except KeyError:
            print(f"Error: Target column '{self.target_column_name}' not found in data.")
            return None
        
        # Create a new column order with the target column first
        cols = [self.target_column_name] + [col for col in df_hourly.columns if col != self.target_column_name]
        df_hourly = df_hourly[cols]
        # Now, the target_column_index is always 0
        self.target_column_index = 0 
        
        return df_hourly

    def _split_and_scale(self, df):
        """
        Private method to split into train/val/test and fit/transform the scaler.
        """
        print("Step 3/5: Splitting data and applying scaler...")
        
        # We split on a roughly 70-15-15 partition
        train_df = df.loc['2006-12-16':'2009-11-30']
        val_df = df.loc['2009-12-01':'2010-04-30']
        test_df = df.loc['2010-05-01':]
        
        # Fit the scaler only on the training data
        self.scaler.fit(train_df)
        
        # Apply our scaler to every subset
        scaled_train = self.scaler.transform(train_df)
        scaled_val = self.scaler.transform(val_df)
        scaled_test = self.scaler.transform(test_df)
        
        return scaled_train, scaled_val, scaled_test

    def _create_windows(self, data):
        """
        Private method that applies the windowing function.
        Uses class attributes for steps and target index.
        """
        input_steps = self.input_steps
        output_steps = self.output_steps
        
        X, y = [], []
        
        for i in range(len(data) - input_steps - output_steps + 1):
            
            # Input: All features (now including time features)
            input_window = data[i : (i + input_steps)]
            X.append(input_window)
            
            # Output: Target column only (Global_active_power)
            if self.get_all_label_features: 
                output_window = data[(i + input_steps) : (i + input_steps + output_steps), :]
            else:
                output_window = data[(i + input_steps) : (i + input_steps + output_steps), self.target_column_index]
            y.append(output_window)
            
        return np.array(X), np.array(y)

    def load_and_process_data(self):
        """
        This is the main public method to run the entire pipeline.
        """
        df_clean = self._fetch_clean_and_engineer()
        df_hourly = self._resample_and_reorder(df_clean)
        scaled_train, scaled_val, scaled_test = self._split_and_scale(df_hourly)
        
        print("Step 4/5: Creating time-series windows...")
        self.X_train, self.y_train = self._create_windows(scaled_train)
        self.X_val, self.y_val = self._create_windows(scaled_val)
        self.X_test, self.y_test = self._create_windows(scaled_test)
        
        print("Step 5/5: Data processing complete.")
        
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)

    def inverse_transform_predictions(self, predictions):
        """
        Helper to convert scaled predictions back to real values.
        Updated to handle 24-hour sequence predictions.
        """
        target_min = self.scaler.min_[self.target_column_index]
        target_scale = self.scaler.scale_[self.target_column_index]
        
        unscaled_predictions = (predictions - target_min) / target_scale
        
        return unscaled_predictions

if __name__ == '__main__':
    # Test block
    processor = DataProcessor(input_steps=48, output_steps=24)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.load_and_process_data()
    print("\n--- Final Data Shapes ---")
    print(f"X_train shape: {X_train.shape}")