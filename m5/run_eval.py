


from sklearn import preprocessing, metrics
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from datetime import datetime

"""


import util_feat_m5



# df_meta=   col_name, col_type, file_path



def features_generate_file(dir_in, dir_out, my_fun_features) :
	
    # from util_feat_m5  import lag_featrues
    # features_generate_file(dir_in, dir_out, lag_featrues) 
	
	train_df = pd.read_csv( dir_in  + "/sales_train_validation.csv.zip")
	calendar_df = pd.read_csv(dir_in  + "/calendar.csv")
	price_df = pd.read_csv(dir_in  + "/sell_prices.csv") 


	dfnew = my_fun_features(train_df, calendar_df, price_df) :

    dfnew.to_parquet( dir_out +"/mfeaturesXXXX.parquet")



 def features_get_cols(mode="random") :
    cols_cat0 = [  "feat1", "fewat2" ]

    if mode == "random" :
		### Random selection
	    cols_cat = cols_cat0[ np.random.c = hoice( 5, len(cols_cat)  ) ]
	    cols_num = cols_num0[ np.random.c = hoice( 5, len(cols_num)  ) ]
        return cols_cat, col_num

    if mode == "all" :
      pass    

    if mode == "smartway" :
       pass



def run_eval(model, pars={} ) :

    data_pars = {}
    model_pars = {}

    for ii in range(n_experiments) :
		cols_cat, cols_num = features_get_cols()

        df     = load_data(path, cols_cat + cols_num, "train")
        dftest = load_data(path, cols_cat + cols_num, 'test')

		X_train = X_transform( df, cols_num, cols_cat, pars) # select sri
		y_train  = y_transform(df, coly) 

		X_test = X_transform( dftest, cols_num, cols_cat, pars) # select variables   
		y_test  = y_transform(dftest, coly) 



		lgbm = lgb.LGBMRegressor()
        lgbm.fit( X_train, y_train)

		# prediction + metrics
		y_test_pred = lgbm.predict(X_test)
        metric_val = metrics_calc(y_test, y_test_pred)


	    ### Store in metrics :
	    
	    # run_id, feat_name, feat_name_long, feat_type, model_params, metric_name, metric_val 
	    # 3,roling_demand,Mean of the variable estimates,lag_features,params = {"objective" : "poisson","metric" :"rmse","force_row_wise" : True,"learning_rate" : 0.075,
	"sub_row" : 0.75,"bagging_freq" : 1,"lambda_l2" : 0.1,"metric": ["rmse"],'verbosity': 1,'num_iterations' : 250,
	},rmse,1.16548
	    
	    df_metrics['run_id'] = time()
	    df_metrics['cols'].append(  ",".join( cols_num + cols_cat ))
	    df_metrics['metrics_val'].append(metric_val)





def test_old():
	from util_feat_m5  import lag_featrues
	train_df = pd.read_csv("sales_train_validation.csv")
	calendar_df = pd.read_csv("calendar.csv")
	price_df = pd.read_csv("sell_prices.csv")
	sample = pd.read_csv("sample_submission.csv")
	calendar_df["date_dt"] = pd.to_datetime(calendar_df["date"])
	train  = train_df.copy()
	price = price_df.copy()
	calendar = calendar_df.copy()
	Train_data = train.iloc[:,:-56]
	Val_data = train.iloc[:,:-28]

	X_train = lag_featrues(Train_data).iloc[:,5:] # select variables
	y_train = train.iloc[:,-56]
	X_test = lag_featrues(Val_data).iloc[:,5:]
	y_test = train.iloc[:,-28]

	# Create instance
	lgbm = lgb.LGBMRegressor()

	# Training and score
	learning_rate = [0.15, 0.2, 0.25]
	max_depth = [15, 20, 25]

	param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}

	# Fitting
	cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)
	cv_lgbm.fit(X_train, y_train)

	print("Best params:{}".format(cv_lgbm.best_params_))

	# best params
	best_lg = cv_lgbm.best_estimator_

	# prediction
	y_train_pred_lg = best_lg.predict(X_train)
	y_test_pred_lg = best_lg.predict(X_test)

	print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))
	print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred_lg)))

	print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))
	print("R2 score test:{}".format(r2_score(y_test, y_test_pred_lg)))
	#Predict using only variables with an importance of 1 or higher.
	importance = best_lg.feature_importances_

	indices = np.argsort(importance)[::-1]

	# print importance
	importance_df = pd.DataFrame({})
	columns = []
	importance_ = []
	for f in range(X_train.shape[1]):
	    print("%2d) %-*s %.2f" %(f+1, 30, X_train.columns[indices[f]], importance[indices[f]]))
	    col = X_train.columns[indices[f]]
	    imp = importance[indices[f]]
	    columns.append(col)
	    importance_.append(imp)
	importance_df["col_name"] = columns
	importance_df["importance"] = importance_
	importance = best_lg.feature_importances_

	indices = np.argsort(importance)[::-1]

	# importance columns (>0)
	imp_col = importance_df[importance_df["importance"]>0]["col_name"].values

	# Train test split, select by imp_col

	X_train = lag_featrues(Train_data).iloc[:,5:][imp_col] # select variables
	y_train = train.iloc[:,-56]
	X_test = lag_featrues(Val_data).iloc[:,5:][imp_col]
	y_test = train.iloc[:,-28]

	# Create instance
	lgbm = lgb.LGBMRegressor()

	# Training and score
	learning_rate = [0.15, 0.2, 0.25]
	max_depth = [15, 20, 25]

	param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}

	# Fitting
	cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)
	cv_lgbm.fit(X_train, y_train)

	print("Best params:{}".format(cv_lgbm.best_params_))

	# best params
	best_lg = cv_lgbm.best_estimator_

	# prediction
	y_train_pred_lg = best_lg.predict(X_train)
	y_test_pred_lg = best_lg.predict(X_test)

	print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))
	print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred_lg)))

	print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))
	print("R2 score test:{}".format(r2_score(y_test, y_test_pred_lg)))
	run_id=list(range(300))
	df_metrics=pd.DataFrame(run_id,columns=['run_id'])
	df_metrics['feat_name'] = pd.Series(X_train.columns[0:300], index=dataframe.index) 
	df_metrics['feat_type']=df_metrics.feat_name
	df_metrics.replace({'feat_type': r'^lag_.*'}, {'feat_type': 'lag'}, regex=True,inplace=True)
	df_metrics.replace({'feat_type': r'^rolling.*'}, {'feat_type': 'rolling'}, regex=True,inplace=True)
	df_metrics['parameter'] = pd.Series(best_lg, index=dataframe.index) 
	df_metrics['metric_name'] ="MSE" 
	df_metrics['metric_val'] = pd.Series(pred_mse[:300], index=dataframe.index) 
	df_metrics.to_csv("run_eval.csv")
"""




def transform_categorical_features(df):
	nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
	for feature in nan_features:
		df[feature].fillna('unknown', inplace = True)
	
	categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
	for feature in categorical_cols:
		encoder = preprocessing.LabelEncoder()
		df[feature] = encoder.fit_transform(df[feature].astype(str))

	return df


# def get_parquet_file_name(feature_set_name, max_rows):
# 	return f'features_{feature_set_name}_{max_rows}.parquet'



# def create_and_save_features(max_rows, feature_set_names):
# 	for feature_set_name in feature_set_names:
# 		df_feature = pd.DataFrame()
# 		merged_df = prepare_raw_merged_df(max_rows)
# 		if feature_set_name == "set1":
# 			df_feature = create_set1_features(merged_df)
# 		elif feature_set_name == "set2":
# 			df_feature = create_set2_features(merged_df)
# 		df_feature.to_parquet(get_parquet_file_name(feature_set_name, max_rows))
# 		print(f'Saving data set with {max_rows} rows named {feature_set_name}')





def add_time_features(df):
	df['date'] = pd.to_datetime(df['date'])
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month
	df['week'] = df['date'].dt.week
	df['day'] = df['date'].dt.day
	df['dayofweek'] = df['date'].dt.dayofweek
	return df


def prepare_raw_merged_df(max_rows):
	df_sales_train = pd.read_csv("sales_train_evaluation.csv/sales_train_evaluation.csv")
	df_calendar = pd.read_csv("calendar.csv")
	df_sales_validation = pd.read_csv("sales_train_validation.csv/sales_train_validation.csv")
	df_sell_price = pd.read_csv("sell_prices.csv/sell_prices.csv")
	df_submission = pd.read_csv("sample_submission.csv/sample_submission.csv")

	df_sales_validation_melt = pd.melt(df_sales_validation[0:max_rows], id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	validation_rows = [row for row in df_submission['id'] if 'validation' in row]
	evaluation_rows = [row for row in df_submission['id'] if 'evaluation' in row]
	df_submission_validation = df_submission[df_submission['id'].isin(validation_rows)][0:max_rows]
	df_submission_evaluation = df_submission[df_submission['id'].isin(evaluation_rows)][0:max_rows]
	
	df_product = df_sales_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']][1:1000].drop_duplicates()
    
	df_submission_validation = df_submission_validation.merge(df_product, how = 'left', on = 'id')
	df_submission_evaluation = df_submission_evaluation.merge(df_product, how = 'left', on = 'id')

	df_submission_validation = pd.melt(df_submission_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	df_submission_evaluation = pd.melt(df_submission_evaluation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    
	df_sales_validation_melt['part'] = 'train'
	df_submission_validation['part'] = 'test1'
	df_submission_evaluation['part'] = 'test2'
    
	merged_df = pd.concat([df_sales_validation_melt, df_submission_validation, df_submission_evaluation], axis = 0)
	df_calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
	merged_df = pd.merge(merged_df, df_calendar, how = 'left', left_on = ['day'], right_on = ['d'])
	merged_df = merged_df.merge(df_sell_price, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

	merged_df = transform_categorical_features(merged_df)
	merged_df = add_time_features(merged_df)

	return merged_df


def features_get_cols(mode = "random"):
	categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2' , 'id_encode',]
	numerical_cols = ['snap_TX',  'sell_price', 'week', 'snap_CA', 'month', 'snap_WI', 'dayofweek', 'year']

	cols_cat = []
	cols_num = []

	if mode == "random":
		cols_cat = [categorical_cols[i] for i in np.random.choice(len(categorical_cols), 5, replace = False)]
		cols_num = [numerical_cols[i] for i in np.random.choice(len(numerical_cols), 5, replace = False) ]

	if mode == "all":
		cols_cat = categorical_cols
		cols_num = numerical_cols

	if mode == "smartway":
		cols_cat = categorical_cols
		cols_num = numerical_cols
		# TODO: Need to update

	return cols_cat, cols_num



def load_data(path, selected_cols, part):
	selected_cols = ['demand', 'date', 'part'] + selected_cols

	merged_df = pd.read_parquet(path, columns = selected_cols)
	merged_df = merged_df[merged_df['part'] == part].sort_values('date')
	return merged_df.drop(['part'], axis=1)


def X_transform(df, selected_cols):
	X_df = df.drop(['demand', 'date'], axis =1)
	return X_df


def Y_transform(df, selected_col):
	Y_df= df[selected_col]
	return Y_df


def run_eval(max_rows = None, n_experiments = 3):
	model_params = {'num_leaves': 555,
          'min_child_weight': 0.034,
          'feature_fraction': 0.379,
          'bagging_fraction': 0.418,
          'min_data_in_leaf': 106,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.3899,
          'reg_lambda': 0.648,
          'random_state': 222,
         }

	dict_metrics = {'run_id' : [], 'cols' : [], 'metric_name': [], 'model_params': [], 'metrics_val' : []}

	for ii in range(n_experiments):
		cols_cat, cols_num = features_get_cols()
		df_train = load_data('features_set1_100.parquet', cols_cat + cols_num, 'train')
		df_test = load_data('features_set1_100.parquet', cols_cat + cols_num, 'test1')
		print('Features loaded')

		X_train = X_transform(df_train, cols_cat + cols_num)
		Y_train = Y_transform(df_train, 'demand')
		X_test = X_transform(df_test, cols_cat + cols_num)
		Y_test = Y_transform(df_test, 'demand')

		# preparing split
		n_fold = 3
		folds = TimeSeriesSplit(n_splits=n_fold)
		splits = folds.split(X_train, Y_train)

		Y_test = np.zeros(X_test.shape[0])

		for fold_n, (train_index, valid_index) in enumerate(splits):
			print('Fold:',fold_n+1)
			X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
			Y_train_fold, Y_valid_fold = Y_train.iloc[train_index], Y_train.iloc[valid_index]
			dtrain = lgb.Dataset(X_train_fold, label=Y_train_fold)
			dvalid = lgb.Dataset(X_valid_fold, label=Y_valid_fold)
			clf = lgb.train(model_params, dtrain, 2500, valid_sets = [dtrain, dvalid],early_stopping_rounds = 50, verbose_eval=100)
			Y_valid_fold_pred = clf.predict(X_valid_fold,num_iteration=clf.best_iteration)
			val_score = np.sqrt(metrics.mean_squared_error(Y_valid_fold_pred, Y_valid_fold))
			print(f'val rmse score is {val_score}')

			Y_test += clf.predict(X_test, num_iteration=clf.best_iteration)/n_fold

			dict_metrics['run_id'].append(datetime.now())
			dict_metrics['cols'].append(";".join(X_train.columns.tolist()))
			dict_metrics['model_params'].append(model_params)
			dict_metrics['metric_name'].append('rmse')
			dict_metrics['metrics_val'].append(val_score)

	df_metrics = pd.DataFrame.from_dict(dict_metrics)
	print("****************************")
	print("        DF metrics          ")
	print("****************************")
	print(df_metrics)
	df_metrics.to_csv("df_metrics.csv")





if __name__ == "__main__":
	# create_and_save_features(100, ["set1", "set2"])
	run_eval(100)
