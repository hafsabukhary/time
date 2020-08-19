
"""

Packages to use :

tsfresh

tsfel https://tsfel.readthedocs.io/en/latest/

sktime

feature tools : https://docs.featuretools.com/en/stable/automated_feature_engineering/handling_time.html

Cesium http://cesium-ml.org/docs/feature_table.html

Feature Tools for advacned fewatures `https://github.com/Featuretools/predict-remaining-useful-life/blob/master/Advanced%20Featuretools%20RUL.ipynb


"""

import pandas as pd
import tsfresh




def features_time_basic(dfraw, fname):
	df = copy.deepcopy(dfraw)
	df['date'] = pd.to_datetime(df['date'])
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month
	df['week'] = df['date'].dt.week
	df['day'] = df['date'].dt.day
	df['dayofweek'] = df['date'].dt.dayofweek	
	df.to_parquet(fname)


    
    
def features_lag(df, fname):
    out_df = df[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    ###############################################################################
    # day lag 29~57 day and last year's day lag 1~28 day 
    day_lag = df.iloc[:,-28:]
    day_year_lag = df.iloc[:,-393:-365]
    day_lag.columns = [str("lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    day_year_lag.columns = [str("lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    # Rolling mean(3) and (7) and (28) and (84) 29~57 day and last year's day lag 1~28 day 
    rolling_3 = df.iloc[:,-730:].T.rolling(3).mean().T.iloc[:,-28:]
    rolling_3.columns = [str("rolling3_lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    rolling_3_year = df.iloc[:,-730:].T.rolling(3).mean().T.iloc[:,-393:-365]
    rolling_3_year.columns = [str("rolling3_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_7 = df.iloc[:,-730:].T.rolling(7).mean().T.iloc[:,-28:]
    rolling_7.columns = [str("rolling7_lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    rolling_7_year = df.iloc[:,-730:].T.rolling(7).mean().T.iloc[:,-393:-365]
    rolling_7_year.columns = [str("rolling7_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_28 = df.iloc[:,-730:].T.rolling(28).mean().T.iloc[:,-28:]
    rolling_28.columns = [str("rolling28_lag_{}_day".format(i)) for i in range(29,57)]
    rolling_28_year = df.iloc[:,-730:].T.rolling(28).mean().T.iloc[:,-393:-365]
    rolling_28_year.columns = [str("rolling28_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_84 = df.iloc[:,-730:].T.rolling(84).mean().T.iloc[:,-28:]
    rolling_84.columns = [str("rolling84_lag_{}_day".format(i)) for i in range(29,57)]
    rolling_84_year = df.iloc[:,-730:].T.rolling(84).mean().T.iloc[:,-393:-365]
    rolling_84_year.columns = [str("rolling84_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    # monthly lag 1~18 month
    month_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly = df.iloc[:,-28*i:].T.sum().T
            month_lag["monthly_lag_{}_month".format(i)] = monthly
        else:
            monthly = df.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_lag["monthly_lag_{}_month".format(i)] = monthly

    # combine day lag and monthly lag
    out_df = pd.concat([out_df, day_lag], axis=1)
    out_df = pd.concat([out_df, day_year_lag], axis=1)
    out_df = pd.concat([out_df, rolling_3], axis=1)
    out_df = pd.concat([out_df, rolling_3_year], axis=1)
    out_df = pd.concat([out_df, rolling_7], axis=1)
    out_df = pd.concat([out_df, rolling_7_year], axis=1)
    out_df = pd.concat([out_df, rolling_28], axis=1)
    out_df = pd.concat([out_df, rolling_28_year], axis=1)
    out_df = pd.concat([out_df, rolling_84], axis=1)
    out_df = pd.concat([out_df, rolling_84_year], axis=1)
    out_df = pd.concat([out_df, month_lag], axis=1)

    ###############################################################################
    # dept_id
    group_dept = df.groupby("dept_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    dept_day_lag = group_dept.iloc[:,-28:]
    dept_day_year_lag = group_dept.iloc[:,-393:-365]
    dept_day_lag.columns = [str("dept_lag_{}_day".format(i)) for i in range(29,57)]
    dept_day_year_lag.columns = [str("dept_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_dept_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_dept = group_dept.iloc[:,-28*i:].T.sum().T
            month_dept_lag["dept_monthly_lag_{}_month".format(i)] = monthly_dept
        elif i >= 7 and i < 13:
            continue
        else:
            monthly = group_dept.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_dept_lag["dept_monthly_lag_{}_month".format(i)] = monthly_dept
    # combine out df
    out_df = pd.merge(out_df, dept_day_lag, left_on="dept_id", right_index=True, how="left")
    out_df = pd.merge(out_df, dept_day_year_lag, left_on="dept_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_dept_lag, left_on="dept_id", right_index=True, how="left")

    ###############################################################################       
    # cat_id
    group_cat = df.groupby("cat_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    cat_day_lag = group_cat.iloc[:,-28:]
    cat_day_year_lag = group_cat.iloc[:,-393:-365]
    cat_day_lag.columns = [str("cat_lag_{}_day".format(i)) for i in range(29,57)]
    cat_day_year_lag.columns = [str("cat_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_cat_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_cat = group_cat.iloc[:,-28*i:].T.sum().T
            month_cat_lag["cat_monthly_lag_{}_month".format(i)] = monthly_cat
        elif i >= 7 and i < 13:
            continue
        else:
            monthly_cat = group_cat.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_cat_lag["dept_monthly_lag_{}_month".format(i)] = monthly_cat

    # combine out df
    out_df = pd.merge(out_df, cat_day_lag, left_on="cat_id", right_index=True, how="left")
    out_df = pd.merge(out_df, cat_day_year_lag, left_on="cat_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_cat_lag, left_on="cat_id", right_index=True, how="left")

    ###############################################################################
    # store_id
    group_store = df.groupby("store_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    store_day_lag = group_store.iloc[:,-28:]
    store_day_year_lag = group_store.iloc[:,-393:-365]
    store_day_lag.columns = [str("store_lag_{}_day".format(i)) for i in range(29,57)]
    store_day_year_lag.columns = [str("store_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_store_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_store = group_store.iloc[:,-28*i:].T.sum().T
            month_store_lag["store_monthly_lag_{}_month".format(i)] = monthly_store
        elif i >= 7 and i <13:
            continue
        else:
            monthly_store = group_store.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_store_lag["store_monthly_lag_{}_month".format(i)] = monthly_store

    # combine out df
    out_df = pd.merge(out_df, store_day_lag, left_on="store_id", right_index=True, how="left")
    out_df = pd.merge(out_df, store_day_year_lag, left_on="store_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_store_lag, left_on="store_id", right_index=True, how="left")

    ###############################################################################
    # state_id
    group_state = df.groupby("state_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    state_day_lag = group_state.iloc[:,-28:]
    state_day_year_lag = group_state.iloc[:,-393:-365]
    state_day_lag.columns = [str("state_lag_{}_day".format(i)) for i in range(29,57)]
    state_day_year_lag.columns = [str("state_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_state_lag = pd.DataFrame({})
    for i in range(1,13):
        if i == 1:
            monthly_state = group_state.iloc[:,-28*i:].T.sum().T
            month_state_lag["state_monthly_lag_{}_month".format(i)] = monthly_state
        elif i >= 7 and i < 13:
            continue
        else:
            monthly_state = group_state.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_state_lag["state_monthly_lag_{}_month".format(i)] = monthly_state

    # combine out df
    out_df = pd.merge(out_df, state_day_lag, left_on="state_id", right_index=True, how="left")
    out_df = pd.merge(out_df, state_day_year_lag, left_on="state_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_state_lag, left_on="state_id", right_index=True, how="left")

    ###############################################################################
    # category flag
    col_list = ['dept_id', 'cat_id', 'store_id', 'state_id']

    df_cate_oh = pd.DataFrame({})
    for i in col_list:
        df_oh = pd.get_dummies(df[i])
        df_cate_oh = pd.concat([df_cate_oh, df_oh], axis=1)

    out_df = pd.concat([out_df, df_cate_oh], axis=1)

    out_df.to_parquet(fname) 
    # return out_df
  
  
  
  
  
  
  
  
  
  
  
