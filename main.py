import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
import statistics
import time
import warnings

warnings.filterwarnings('ignore')

def get_season():
  return 2024

def get_league():
  return 'epl'

def smape(y_true, y_pred):
  return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def get_modeling_table():
  # READ INSTRUCTIONS IN modeling_table_download_instructions.txt FILE
  # THIS WILL NOT WORK UNLESS YOU DOWNLOAD THE MODELING TABLE CSV FILE
  df = pd.read_csv('modeling_table.csv')
  return get_position_group(df)

def get_target_features():
  return pd.read_excel('features.xlsx', sheet_name = 'target_features')['feature'].tolist()

def get_position_group(df):
  conditions = [
      df['main_position'].isin(['FW', 'RW', 'LW']),
      df['main_position'].isin(['AM', 'DM', 'CM', 'LM', 'RM']),
      df['main_position'].isin(['CB', 'LB', 'RB', 'WB']),
      df['main_position'] == 'GK'
  ]
  choices = ['FWD', 'MID', 'DEF', 'GK']

  df['position_group'] = np.select(conditions, choices, default=None)

  return df

def adjust_skewness():

  binary_cols = [
      'is_home', 'yellow_card', 'second_yellow_card', 
      'red_card', '60min_clean_sheet', '90min_clean_sheet'
  ]

  categorical_cols = [
      'main_position', 'position_group'
  ]

  numeric_cols = [
      'age_days', 'round', 'team_elo', 
      'opponent_elo', 'team_elo_diff'
  ]

  id_cols = [
      'league', 'season', 'date', 'game_id', 
      'team', 'opponent', 'player', 'position', 'key'
  ]

  assigned_cols = binary_cols + categorical_cols + numeric_cols + id_cols

  log_cols, sqrt_cols = [], []
  for col in df.columns:
      if col in assigned_cols:
          continue
      try:
          skew = abs(df[col].skew())
          if skew > 1:
              log_cols.append(col)
          elif 0.5 < skew <= 1:
              sqrt_cols.append(col)
          else:
              numeric_cols.append(col)
      except Exception as e:
          print(f"Error processing column {col}: {e}")

  return log_cols, sqrt_cols, numeric_cols, binary_cols, categorical_cols

def get_train_test_split(df):
  test_round = df[(df['season'] == get_season()) & (df['league'] == get_league())]['round'].max()

  test = df[(df['season'] == get_season()) & (df['round'] == test_round)]
  train = df[~df.index.isin(test.index)]  

  test = test[test['league'] == get_league()]

  train = train[train['is_player_modeling_table'] == 1]
  test = test[test['is_player_modeling_table'] == 1]

  return train, test

def train_and_test_model(train, test):
  test_pred_df = test.copy()[['league', 'season', 'date', 'team', 'opponent', 'player', 'main_position', 'position_group']]

  for outcome_variable in get_target_features():
    print(outcome_variable)

    log_cols, sqrt_cols, numeric_cols, binary_cols, categorical_cols = adjust_skewness()

    ml_cols = ['main_position', 'position_group', 'round', 'age_days', 
        'is_home', 'team_elo', 'opponent_elo', 'team_elo_diff', 'minutes']
    ml_cols += [col for col in df.columns if ('prev_' in col)]

    if outcome_variable == 'minutes':
      ml_cols.remove('minutes')

    log_cols_filtered = [i for i in log_cols if i in ml_cols]
    sqrt_cols_filtered = [i for i in sqrt_cols if i in ml_cols]
    numeric_cols_filtered = [i for i in numeric_cols if i in ml_cols]
    binary_cols_filtered = [i for i in binary_cols if i in ml_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('log', FunctionTransformer(np.log1p, validate=False), log_cols_filtered),
            ('sqrt', FunctionTransformer(np.sqrt, validate=False), sqrt_cols_filtered),
            ('scaler', StandardScaler(), numeric_cols_filtered),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('binary', 'passthrough', binary_cols_filtered)
        ],
        remainder='passthrough'
    )

    X_train = train[ml_cols]
    y_train = train[outcome_variable]
    X_test = test[ml_cols]
    y_test = test[outcome_variable]

    if outcome_variable not in binary_cols:
      pipeline = Pipeline([
              ('preprocessor', preprocessor),
              ('feature_selector', SelectFromModel(lgb.LGBMRegressor(), threshold="mean", max_features=25)),
              ('model', lgb.LGBMRegressor())
          ])
    else: 
      pipeline = Pipeline([
              ('preprocessor', preprocessor),
              ('feature_selector', SelectFromModel(lgb.LGBMClassifier(), threshold="mean", max_features=25)),
              ('model', lgb.LGBMClassifier())
          ])
      
    pipeline.fit(X_train, y_train)

    if outcome_variable not in binary_cols:
      y_pred_train = pipeline.predict(X_train)
      y_pred_test = pipeline.predict(X_test)
    else: 
      y_pred_train = pipeline.predict_proba(X_train)[:, 1]
      y_pred_test = pipeline.predict_proba(X_test)[:, 1]

    y_pred_train = [round(max(0, num), 2) for num in y_pred_train]
    y_pred_test = [round(max(0, num), 2) for num in y_pred_test]

    train_r2 = metrics.r2_score(y_train, y_pred_train)
    test_r2 = metrics.r2_score(y_test, y_pred_test)
    test_mean = statistics.mean(y_test)
    test_mae = metrics.mean_absolute_error(y_test, y_pred_test)
    test_smape = smape(y_test, y_pred_test)
    test_mse = metrics.mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    try: test_mean_poisson_deviance = metrics.mean_poisson_deviance([0.00001 if num == 0 
            else num for num in y_test.tolist()], [0.00001 if num == 0 else num for num in y_pred_test])
    except: test_mean_poisson_deviance = None
    test_mbe = np.mean(y_pred_test - y_test)

    print(f"Training R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test SMAPE: {test_smape:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    try: print(f"Test Mean Poisson Deviance: {test_mean_poisson_deviance:.4f}")
    except: pass
    print(f"Test MBE: {test_mbe:.4f}")

    test_pred_df[outcome_variable] = test[outcome_variable]
    test_pred_df[outcome_variable + '_pred'] = y_pred_test    
    test_pred_df[outcome_variable + '_train_r2_score'] = train_r2
    test_pred_df[outcome_variable + '_r2_score'] = test_r2
    test_pred_df[outcome_variable + '_mean'] = test_mean
    test_pred_df[outcome_variable + '_mae'] = test_mae
    test_pred_df[outcome_variable + '_smape'] = test_smape
    test_pred_df[outcome_variable + '_mse'] = test_mse
    test_pred_df[outcome_variable + '_rmse'] = test_rmse
    test_pred_df[outcome_variable + '_mean_poisson_deviance'] = test_mean_poisson_deviance
    test_pred_df[outcome_variable + '_mbe'] = test_mbe

    if outcome_variable == 'minutes':
      test['minutes'] = y_pred_test

  test_round = test['round'].max()
  test_pred_df.to_csv(f'test_predictions_round{test_round}.csv', index=False)

  return test_pred_df

df = get_modeling_table()
train, test = get_train_test_split(df)
test_pred_df = train_and_test_model(train, test)
