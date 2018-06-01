import pandas as pd
import quandl
import matplotlib.pyplot as plt
from matplotlib import style
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import pickle

style.use('ggplot')

# df = quandl.get('BCHAIN/TRVOU')
df = pd.read_csv('out.csv')
# df.to_csv('out.csv')
print(df.tail())

# filter the columns we want
df = df [['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume','Date']]

# get the high and low values as %
df['HL_PCT'] = (df ['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] *100.0
# get the percentage of change
df['PCT_change'] = (df ['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] *100.0

df = df [['Adj. Close','HL_PCT' ,'PCT_change','Adj. Volume']]
forecast_col = 'Adj. Close'

# Fill the Nan with -99999
df.fillna(-99999,inplace=True)

# get the 10% length of dataframe
forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# remove NaN
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1) # parallel
clf.fit(X_train,y_train)

# save de results to don't train every run
# with open ('linearregression.pickle','wb') as f:
#     pickle.dump(clf,f)

pickle_in = open ('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
#
# train linear regression
accuracy = clf.score(X_test,y_test)

forecast_set = clf.predict(X_lately)
print( accuracy)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date#.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

