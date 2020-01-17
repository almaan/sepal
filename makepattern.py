#/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def pattern1(x,y,r = 5,c = 15,oval = 20):
    ss = (x-c)**2 + (y-c)**2
    ret = np.ones(x.shape[0])*5
    ret[np.abs(r**2 - ss) < 8] = oval 

    return ret

def pattern2(x,y,bx,by,):


xx = np.arange(0,30)
yy = np.arange(0,30)
ngenes = 100


x,y = np.meshgrid(xx,yy)
x = x.flatten()
y = y.flatten()
v = fun(x,y)
v = np.random.normal(v,2)
v[v < 0] = 0.0
v = v.round()

crd = np.array([str(x) + 'x' + str(y) for x,y in zip(x,y)])
mat = np.zeros((x.shape[0],ngenes))
mat[:,0] = v

plt.scatter(x,y,c = v, s = 80)
plt.show()

for p in range(1,ngenes):

    f = np.random.choice(np.arange(1,4)).round(0).astype(int)
    for i in range(f):
        mat[:,p] += np.random.permutation(v)

names = pd.Index(["Gene_" + str(s+1) for s in range(ngenes)])

df = pd.DataFrame(mat,index = pd.Index(crd),columns = names)

df.to_csv("/tmp/test.tsv",sep = '\t',header = True,index = True)
