import pandas as pd
import numpy as np
from glob import glob
from os.path import basename

def p_calc(c12, c1, c2, N):
    #insert your code here
    #p = c2/N #Doesn't need to be calculated for every word.
    p1 = c12/c1
    p2 = (c2-c12)/(N-c1)
    return p1, p2

def log_L(k,n,x):
    #insert your code here
    return (np.log(x)*k)+(np.log(1-x)*(n-k))

def log_likelihood(c1, c2, c12, p, p1, p2, N):
    #insert your code here.
    
    e1 = log_L(c12, c1, p).replace([np.inf, -np.inf, np.nan], 0)
    e2 = log_L(c2 - c12, N-c1, p).replace([np.inf, -np.inf, np.nan], 0)
    e3 = log_L(c12, c1, p1).replace([np.inf, -np.inf, np.nan], 0)
    e4 = log_L(c2-c12, N-c1, p2).replace([np.inf, -np.inf, np.nan], 0)
    return -2*(e1 + e2 - e3 - e4)

def calculation():
    for filename in ['/media/matt/DATA/GitHub/ESU_2014/Notebooks/Data/blake-songs.txt.cooc.pickle']:
        print(filename)
        df = pd.read_pickle(filename)
        c2 = df.sum()/8
        N = df.values.sum()/8
        p = c2/N
        LL_df = pd.DataFrame(index = df.index, columns = df.columns)
        counter = 0
        # initializing an empty DataFrame like this will reserve the necessary memory to build it
        # But you shouldn't initialize until you are ready to build it, as we are here.
        for t in df.index:
            if counter % 100 == 0:
                print('Now calculating row %s of %s' % (counter, len(df)))
            p1, p2 = p_calc(df.ix[t], df.ix[t].sum()/8, c2, N)
            LL_df.ix[t] = log_likelihood(df.ix[t].sum()/8, c2, df.ix[t], p, p1, p2, N)
            counter +=1
        LL_df.to_pickle('/media/matt/DATA/GitHub/ESU_2014/Notebooks/Data/%s.LL.pickle' % (basename(filename)[:-6]))
