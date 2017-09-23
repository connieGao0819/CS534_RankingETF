# coding: utf-8

# In[5]:

####Response is log(adjust-price)
import csv
import math
tmp,res = [],[]
len_predic = 22 
n = 0

with open('/Users/macbook/Desktop/534 AI/Project/XLK.csv','r') as csvfile:
    etf = csv.reader(csvfile, delimiter = ",")
    for row in etf:
        if n > 0: tmp += [row[6]]
        n += 1
with open('/Users/macbook/Desktop/534 AI/Project/XLK_Output.csv','w',newline = "") as csv_new:
    etf_n = csv.writer(csv_new,delimiter = ",")
    for idx,inner in enumerate(tmp):
        if idx+len_predic+1 >= n-1: break
        inner = float(inner)
        res.append(math.log10(inner))
        for i in range(1,len_predic+1):
            res.append(math.log10(float(tmp[idx+i]))-math.log10(float(tmp[idx+i+1])))
        etf_n.writerow(res)
        res = []


# In[4]:

#####Response is log(adj-prive)[i] - log(adj-price)[i+1] (return)###
import csv
import math
tmp,res = [],[]
len_predic = 22 
n = 0

with open('/Users/macbook/Desktop/534 AI/Project/XLP.csv','r') as csvfile:
    etf = csv.reader(csvfile, delimiter = ",")
    for row in etf:
        if n > 0: tmp += [row[6]]
        n += 1
with open('/Users/macbook/Desktop/534 AI/Project/XLP_Output.csv','w',newline = "") as csv_new:
    etf_n = csv.writer(csv_new,delimiter = ",")
    for idx,inner in enumerate(tmp):
        if idx+len_predic+1 >= n-1: break
        inner = float(inner)
        res.append(math.log10(inner)-math.log10(float(tmp[idx+1])))
        for i in range(1,len_predic+1):
            res.append(math.log10(float(tmp[idx+i]))-math.log10(float(tmp[idx+i+1])))
        etf_n.writerow(res)
        res = []


# In[ ]:




