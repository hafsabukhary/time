

import pandas as pd



def feat_min(df, cols, verbose=False, **kw) :  
  df2 = pd.DataFrame()
  for col in cols :
     df2[col + "_min"] =   df[col].min()
   return df2
   
   


