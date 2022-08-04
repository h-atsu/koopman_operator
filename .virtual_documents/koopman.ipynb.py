import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df1 = pd.read_csv('data/data1.csv', header=None)


df2 = pd.read_csv('data/data2.csv', header=None)


df1.head()


df2.head()


#df2.drop(0, axis = 1,inplace=True)


for i in range(df1.shape[1]):
    df1.iloc[:,i].plot(marker = 'o', markersize = 3)
plt.title('trajectry')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')    


for i in range(df2.shape[1]):
    if i == 0: continue
    df2.iloc[:,i].plot(marker = 'o',markersize = 5)
plt.title('trajectry')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')


def KMD(X, yn):
    """
    observation vector
    X = [y1, y2, ..., yn-1]
    
    output 
    Koopman eigenvalue, Koopman eigen vector
    """
    c = np.linalg.pinv(X.T@X)@X.T@yn #rank落ちしている場合を考慮してムーアペンローズの擬似逆行列を使う, 
    r = yn - X@c #剰余ベクトル
    n = X.shape[1]  #Koopman eigenvalueの個数
    C = np.zeros((n,n)) 
    C[:,-1] = c
    C[np.arange(n-1)+1,np.arange(n-1)] = 1   #コンパニオン行列
    lam, _ = np.linalg.eig(C)                #Koopman eigenvalue
    T = np.vander(lam,increasing=True)     #vandermond matrix
    V = X@np.linalg.pinv(T)                   #Koopman eigen vector
    return lam, V, T, r


Y = df1.values.T


yn = Y[:,-1]
X = Y[:,:-1]


yn.shape


Y.shape


X.shape


lam, V, T, r = KMD(X,yn)


fig = plt.figure(figsize=(5,5))
x,y= [],[]
for j in np.linspace(0, 2 * np.pi, 1000):
      x.append(math.sin(j))
      y.append(math.cos(j))
plt.plot(x,y)    
for l in lam:
    plt.scatter(l.real, l.imag)

plt.title('distribution of eigval')    
plt.grid()
plt.xlim(-2,2)    
plt.ylim(-2,2)


np.argwhere(abs(lam) > 1)


T.shape


Tp = np.concatenate([T, (lam**(T.shape[0])).reshape(-1,1)], axis = 1)
yhat = V@Tp
yhat[:,-1] += r
for i in range(yhat.shape[0]):
    plt.plot(yhat[i].real, marker = 'o', markersize = 3)
plt.title('trajectry')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')


Tp = np.concatenate([T, (lam**(T.shape[0])).reshape(-1,1)], axis = 1)
yhat = V[:,(36,37)]@Tp[(36,37),:]
yhat[:,-1] += r
for i in range(yhat.shape[0]):
    plt.plot(yhat[i].real, marker = 'o', markersize = 3)
plt.title('trajectry')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')


Y = df2.values.T


yn = Y[:,-1]
X = Y[:,:-1]


yn.shape


Y.shape


X.shape


lam, V, T, r = KMD(X,yn)


fig = plt.figure(figsize=(5,5))
label = np.arange(1,lam.size+1)
x,y= [],[]
for j in np.linspace(0, 2 * np.pi, 1000):
      x.append(math.sin(j))
      y.append(math.cos(j))
plt.plot(x,y)    
for i,l in enumerate(lam):
    plt.scatter(l.real, l.imag)
    plt.text(l.real, l.imag, label[i], fontsize=12)

plt.title('distribution of eigval')    
plt.grid()
plt.xlim(-6,6)    
plt.ylim(-6,6)


T = np.vander(lam,increasing=True)
yhat = V@T


Tp = np.concatenate([T, (lam**(T.shape[0])).reshape(-1,1)], axis = 1)
yhat = V@Tp
yhat[:,-1] += r
for i in range(yhat.shape[0]):
    if i == 0:continue
    plt.plot(yhat[i].real, marker = 'o', markersize = 3)
plt.title('trajectry')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')


Tp = np.concatenate([T, (lam**(T.shape[0])).reshape(-1,1)], axis = 1)
yhat = V[:,(0,1)]@Tp[(0,1),:]
yhat[:,-1] += r
for i in range(yhat.shape[0]):
    plt.plot(yhat[i,:-1].real, marker = 'o', markersize = 3)
plt.title('MODE {1,2}')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')


Tp = np.concatenate([T, (lam**(T.shape[0])).reshape(-1,1)], axis = 1)
yhat = V[:,(4,5)]@Tp[(4,5),:]
yhat[:,-1] += r
for i in range(yhat.shape[0]):
    plt.plot(yhat[i,:-1].real, marker = 'o', markersize = 3)
plt.title('MODE {1,2}')    
plt.xlabel('Time')
plt.ylabel('power flow[MW]')



