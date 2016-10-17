#coding=utf-8
import copy
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
lx = 5000
ly = 5000
vstd = 1000000#distance variance
vtstd = 50000000#time variance
num_p = 1000#num of passengers
mean = [lx,ly]
cov = [[vstd,0],[0,vstd]]
texis = np.random.multivariate_normal(mean,cov,num_p)#1000 cars
x,y = texis.T
#plt.plot(x, y, 'x'); plt.axis('equal'); plt.show()

mean1 = [lx,ly,12*60*60]
cov1 = [[vstd,0,0],[0,vstd,0],[0,0,vtstd]]
passengers = np.random.multivariate_normal(mean1,cov1,num_p)#3000 passengers
passengers = list(passengers)
passengers.sort(key=lambda x:x[2])
#print passengers[0:10]
#x1,y1,_ = passengers
#plt.plot(x1, y1, 'x'); plt.axis('equal'); plt.show()
tmp_texi = copy.deepcopy(texis)
distance = 0
for ele in passengers:
    ele = np.array(tmp_texi.shape[0]*[ele[0:2]])
    tds = np.linalg.norm(ele-tmp_texi,axis=0)
    ind = np.argmin(tds)
    distance += tds[ind]
    np.delete(tmp_texi,ind,axis=0)
print distance

tmp_texi = copy.deepcopy(texis)
distance = 0
for i,ele1 in enumerate(passengers):
    ele1_v = np.array(tmp_texi.shape[0]*[ele1[0:2]])
    tds1 = np.linalg.norm(ele1_v-tmp_texi,axis=0)
    ind1 = np.argmin(tds1)
    try:
        ele2 = passengers[i+1]
    except:
        distance += tds1[ind1]
        break
    ele2_v = np.array(tmp_texi.shape[0]*[ele2[0:2]])
    tds2 = np.linalg.norm(ele2_v-tmp_texi,axis=0)
    ind2 = np.argmin(tds2)

    if ind1 == ind2:
        tds1_t = copy.deepcopy(tds1)
        np.delete(tds1_t,ind1,axis=0)
        ind1_2nd = np.argmin(tds1_t)
        tds2_t = copy.deepcopy(tds2)
        np.delete(tds2_t,ind2,axis=0)
        ind2_2nd = np.argmin(tds2_t)
        strategy1 = tds1[ind1] + tds2_t[ind2_2nd]
        strategy2 = tds1_t[ind1_2nd] + tds2[ind2]
        if strategy1 < strategy2:
            distance += tds1[ind1]
            np.delete(tmp_texi,ind1,axis=0)
        else:
            distance += tds1_t[ind1_2nd]
            if ind1>ind1_2nd:
                np.delete(tmp_texi,ind1_2nd,axis=0)
            else:
                np.delete(tmp_texi,ind1_2nd+1,axis=0)
print distance

    
