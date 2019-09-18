import numpy as np
import matplotlib.pyplot as plt
data=np.array([
    [6.1101,17.592],
[5.5277,9.1302],
[8.5186,13.662],
[7.0032,11.854],
[5.8598,6.8233],
[8.3829,11.886],
[7.4764,4.3483],
[8.5781,12],
[6.4862,6.5987],
[5.0546,3.8166],
[5.7107,3.2522],
 [14.164,15.505]
    ])
m=list(data.shape)[0]


X=data[:,:1]
y=data[:,1:]

theta=np.zeros((2,1))

#theta=np.array([[-1],[2]])

#print(theta.shape)

iterations = 1500;
alpha = 0.01;

def computeCost(X, y, theta):
    m=X.size
    zs=np.ones((m,1))
    #print(X.shape,zs.shape)
    Xtemp=np.concatenate((zs,X),axis=1)
    
    
    x=(theta.T).dot(Xtemp.T)
    #print(x.shape)
    x=x.T
    
    d=x-y
    
    d=d**2
    
    su=np.sum(d,axis=0)
    su=su[0]
    #print(su/(2*m)).shape
    return su/(2*m)
    
def gradientDescent(X, y, theta, alpha, num_iters):
    m=X.size
    J_history =[]
    
    
    zs=np.ones((m,1))
    
    Xtemp=np.concatenate((zs,X),axis=1)
    
    for i in range(num_iters):
        
        x=(theta.T).dot(Xtemp.T)
        
        x=x.T
    
        d=x-y
    
        temp=d*Xtemp
        sumst=np.sum(temp,axis=0)*(alpha/m)
        sumst=np.array([sumst])
        
        # simultaneous update rememeber
        sumst=sumst.T
        
        temp1=theta-sumst
            
    
        theta=temp1
        
        #hypothesis evolution
        if(i==0) or ((i+1)%300==0):
          print("epoch ",i+1,)
          predictions=((theta.T).dot(Xtemp.T)).T
   
          plt.plot(X,y,"gx",X,predictions,"b-")   

          plt.title('Linera Regression Fit')
          plt.xlabel("Area of property->")
          plt.ylabel("Cost of property")
          #plt.legend('DP')
          plt.show()
        
        
        
        
        J_history.append(computeCost(X, y, theta))
        #print('Epoch %d J is:   (%f)'%(i+1,computeCost(X, y, theta) ))
        
    return theta
    
#print(gradientDescent(X, y, theta, alpha, 10))    

thetafinal=gradientDescent(X, y, theta, alpha, iterations)
print("Final plot")
m=X.size
zs=np.ones((m,1))
#print(X.shape,zs.shape)
Xtemp=np.concatenate((zs,X),axis=1)


predictions=((thetafinal.T).dot(Xtemp.T)).T
   
plt.plot(X,y,"gx",X,predictions,"b-")   

plt.title('Linera Regression Fit')
plt.xlabel("Area of property->")
plt.ylabel("Cost of property")
plt.legend('DP')
plt.show()
