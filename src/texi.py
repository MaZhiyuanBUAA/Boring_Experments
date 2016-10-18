#coding=utf-8
import copy
import numpy as np
import matplotlib.pyplot as plt
def generateTexisPassengers(seed,lx=5000,ly=5000,sigma1=4e3,sigma2=2.5*3600,num_orders=1000):
    #texis:[(x,y)]
    #passengers:[(x,y,t)]
    np.random.seed(seed)
    mean = [lx,ly]
    cov = [[sigma1**2,0],[0,sigma1**2]]
    texis = np.random.multivariate_normal(mean,cov,num_orders)#1000 cars
    x,y = texis.T
    #plt.plot(x, y, 'x'); plt.axis('equal'); plt.show()
    
    mean1 = [lx,ly,12*60*60]#midi
    cov1 = [[sigma1**2,0,0],[0,sigma1**2,0],[0,0,sigma2**2]]
    passengers = np.random.multivariate_normal(mean1,cov1,num_orders)#3000 passengers
    passengers = list(passengers)
    passengers.sort(key=lambda x:x[2])
    return texis,passengers
#print passengers[0:10]
#x1,y1,_ = passengers
#plt.plot(x1, y1, 'x'); plt.axis('equal'); plt.show()
def miniDist(texis,passengers):
    tmp_texi = copy.deepcopy(texis)
    distance = 0
    for ele in passengers:
        ele = np.array(tmp_texi.shape[0]*[ele[0:2]])
        tds = np.linalg.norm(ele-tmp_texi,axis=0)
        ind = np.argmin(tds)
        distance += tds[ind]
        np.delete(tmp_texi,ind,axis=0)
    return distance

def bigramDist(texis,passengers):
    tmp_texi = copy.deepcopy(texis)
    distance = 0
    time_cost = 0
    for i,ele1 in enumerate(passengers):
        ele1_v = np.array(tmp_texi.shape[0]*[ele1[0:2]])
        tds1 = np.linalg.norm(ele1_v-tmp_texi,axis=1)
        ind1 = np.argmin(tds1)
        try:
            ele2 = passengers[i+1]
        except:
            distance += tds1[ind1]
            break
        ele2_v = np.array(tmp_texi.shape[0]*[ele2[0:2]])
        tds2 = np.linalg.norm(ele2_v-tmp_texi,axis=1)
        ind2 = np.argmin(tds2)
    
        if ind1 == ind2:
            tds1_t = np.delete(tds1,ind1,axis=0)
            ind1_2nd = np.argmin(tds1_t)
            tds2_t = np.delete(tds2,ind2,axis=0)
            ind2_2nd = np.argmin(tds2_t)
            strategy1 = tds1[ind1] + tds2_t[ind2_2nd]
            strategy2 = tds1_t[ind1_2nd] + tds2[ind2]
            if strategy1 < strategy2:
                distance += tds1[ind1]
                tmp_texi = np.delete(tmp_texi,ind1,axis=0)
            else:
                distance += tds1_t[ind1_2nd]
                if ind1>ind1_2nd:
                    tmp_texi = np.delete(tmp_texi,ind1_2nd,axis=0)
                else:
                    tmp_texi = np.delete(tmp_texi,ind1_2nd+1,axis=0)
        else:
            distance += tds1[ind1]
            tmp_texi = np.delete(tmp_texi,ind1,axis=0)
        time_cost += ele2[2] - ele1[2]
    return distance,time_cost

def arg2min(a):
    if not isinstance(a,np.ndarray):
        return IOError
    else:
        ind1 = np.argmin(a)
        a_tmp = np.delete(a,ind1,axis=0)
        ind2 = np.argmin(a_tmp)
        if ind1 > ind2:
            ind2 += 1
        return ind1,ind2
def buildSetforOneDay(texis,passengers):
    data = []#(X,Y):((x1,y1,x2,y2,xp,yp,t),(0,1))
    tmp_texi = copy.deepcopy(texis)
    for i in range(len(passengers)-1):
        ele1 = passengers[i]
        ele1_v = np.array(tmp_texi.shape[0]*[ele1[0:2]])
        tds1 = np.linalg.norm(ele1_v-tmp_texi,axis=1)
        ind11,ind12 = arg2min(tds1)
        #print ind11,ind12
        ele2 = passengers[i+1]
        ele2_v = np.array(tmp_texi.shape[0]*[ele2[0:2]])
        tds2 = np.linalg.norm(ele2_v-tmp_texi,axis=1)
        ind21,ind22 = arg2min(tds2)
        if ind11 == ind21:
            #compare
            strategy1 = tds1[ind11] + tds2[ind22]
            strategy2 = tds1[ind12] + tds2[ind21]
            if strategy1 < strategy2:
                data.append((np.hstack((tmp_texi[ind11],tmp_texi[ind12],ele1)),np.array([1.,0.])))
                tmp_texi = np.delete(tmp_texi,ind11,axis=0)
            else:
                data.append((np.hstack((tmp_texi[ind11],tmp_texi[ind12],ele1)),np.array([0.,1.])))
                tmp_texi = np.delete(tmp_texi,ind12,axis=0)
        else:
            data.append((np.hstack((tmp_texi[ind11],tmp_texi[ind12],ele1)),np.array([1.,0.])))
            tmp_texi = np.delete(tmp_texi,ind11,axis=0)
    return data
def loadData():
    trainSize = 10*1000
    data_x = np.fromfile('../data/data_x.bin',dtype=np.float64)
    data_y = np.fromfile('../data/data_y.bin',dtype=np.float64)
    data_x = np.reshape(data_x,[-1,7])
    #data_x[:,0:6] = np.round(data_x[:,0:6]/10,0)
    #data_x[:,6] = np.round(data_x[:,6]/120,0)
    #print data_x.shape
    data_y = np.reshape(data_y,[-1,2])
    #print data_y[0:10,:]
    trainX = data_x[0:trainSize,:]
    trainY = data_y[0:trainSize,:]
    testX = data_x[trainSize:,:]
    #print testX.shape
    testY = data_y[trainSize:,:]
    return (trainX,trainY,testX,testY)
#a test
#texis,passengers = generateTexisPassengers(seed=1)
#print miniDist(texis, passengers)
#print bigramDist(texis, passengers)

#10 days'data for training,1 day's for testing
def buildSet(days):
    seedTrain = range(days)
    dataSet = []
    for ele in seedTrain:
        print ele
        texis,passengers = generateTexisPassengers(seed=ele)
        dataSet += buildSetforOneDay(texis, passengers)
    data_x,data_y = [],[]
    #data_x:11*1000x7
    #data_y:11*1000x2
    for ele in dataSet:
        data_x.append(ele[0])
        data_y.append(ele[1])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x.tofile('../data/data_x.bin')
    data_y.tofile('../data/data_y.bin')