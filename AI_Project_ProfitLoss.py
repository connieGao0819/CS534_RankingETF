import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# define model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(dim-1, input_dim=dim-1, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

start_time = time.time()
input_file = ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
predict_list,real_list = [],[]
test_size = 22
for file_name in input_file:
    out_path = 'return/Output_ProfitLoss'
    in_path = 'ETF_Newset/'+ file_name + '.csv'
    #load dataset
    dataframe = pandas.read_csv(in_path, delim_whitespace=False, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    dim = len(dataset[0,:])
    X = dataset[:,1:dim]
    Y = dataset[:,0:1]
    X_train,X_test,y_train,y_test = X[test_size+1:,:],X[:test_size+1,:],Y[test_size+1:,:],Y[:test_size+1,:]

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    estimator = baseline_model()
    estimator.fit(X_train,y_train)
    res = estimator.predict(X_test)
    predict_list.append(res.tolist())
    real_list.append(y_test.tolist())
predict_list = np.array(predict_list)
real_list = np.array(real_list)
print("---%s seconds(Total) ---" % (time.time()-start_time))
#print ("Predict:",predict_list)
#print("Real:",real_list)
invest = 100
predict_profit,real_profit = [],[]
for i in range(test_size):
    ##Strategy: Select all the positive return######
    pos_idx = np.where(predict_list[:,i] > 1)[0].tolist()
    #pos_idx = np.where(predict_list[:,i]==np.amax(predict_list[:,i]))
    if len(pos_idx) == 0:
        print("No positive return today: ",i+1)
        predict_profit.append(0)
        real_profit.append(0)
        continue
    tmp_pre = predict_list[pos_idx,i]
    tmp_real = real_list[pos_idx,i]
    predict_profit.append(invest*(np.sum(tmp_pre)))
    real_profit.append(invest*(np.sum(tmp_real)))

#print("Predict_Result:",predict_profit)
#print ("Real_Res:",real_profit)
time = np.arange(1,test_size+1,1)
plt.plot(time,predict_profit,'b',label = 'Predict')
plt.plot(time,real_profit,'g',label = 'Real')
plt.legend()
plt.show()
