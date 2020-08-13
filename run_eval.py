







def run_eval(model, pars={} ) :
  
  
  cols_cat, cols_num = get_features()
  
  
  
  
  
def test():
  import lag_features from util_feat
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
