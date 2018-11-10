import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        #Check if there is more than 1 parameters
        self.vel = -(np.dot(self.lr,grad)) + np.dot(self.beta,self.vel) #set momentum (vel)
        params += self.vel     #Update params 
        
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        N = X.shape[0]   #number of data points 
        hinge_loss_vector = np.zeros(N)    #create a length-n vector 
        wt = self.w.reshape(1,self.w.shape[0])  # transpose weights (w)
        
        for i in range(0,N):
            loss = 1-np.dot(np.dot(y[i],wt),X[i])  #find loss for curent data point
            hinge_loss_vector[i] = max(loss,0)   # save calulated loss only if >=0 else save 0
                   
        return hinge_loss_vector

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        #Set bias to be 0
        weights = self.w
        weights[0] = 0 
        N = X.shape[0]
        m = X.shape[1]

        inner_sum = np.zeros(m)
        hinge_loss_vector = self.hinge_loss(X,y)# find the higneloss for the data points
        
        for i in range(0,N): 
            loss = hinge_loss_vector[i]  #get loss for current data point
            if loss == 0:   # if loss is 0 add nothing to sum
                inner_sum +=0
            else:
                inner_sum += np.dot(y[i],X[i])    #derivative of hingeloss formula wrt w

        grad = weights - (self.c/float(N))*inner_sum        #weights minus the sum
        
        return grad

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1        
        predictions = np.sign(np.dot(X,self.w)) 
        
        return predictions

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        grad = func_grad(w)
        w = optimizer.update_params(w, grad)
        w_history.append(w)

    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    #Initalize Svm and batch
    numOfFeatures = train_data.shape[1]   #get number of feature in data set    
    svm_object = SVM(penalty, numOfFeatures)   # create svm object with learning rate 
    mini_batch = BatchSampler(train_data, train_targets, batchsize)

    #loop iters times and update params for each new batch
    for i in range(0,iters):
        X, y = mini_batch.get_batch()  #get a mini batch
        svm_grad = svm_object.grad(X, y)   #find grad for current batch
        new_weights = optimizer.update_params(svm_object.w, svm_grad)   #use optimizers to update parms
        svm_object.w = new_weights   #save me params 
        
    return svm_object


def train_svm (lr, beta, train_data, train_targets, test_data, test_targets):
    '''Train SVM based on the lr and beta given. Then show accuracy, hinge loss & plot weights 
    '''
    
    #Optimizing SVM
    sgd = GDOptimizer(lr,beta)    
    #optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters)
    svm_object = optimize_svm(train_data,train_targets,1.0, sgd,100,500)
    
    
    #Classifiy  
    train_pred = svm_object.classify(train_data)    
    test_pred = svm_object.classify(test_data)    
    
    
    #Find the accuracy and hinge lossess
    accuracy_train = (train_pred == train_targets).mean()
    accuracy_test = (test_pred == test_targets).mean()
    hinge_train = svm_object.hinge_loss(train_data,train_targets)
    hinge_train = hinge_train.mean()
    hinge_test = svm_object.hinge_loss(test_data,test_targets)
    hinge_test = hinge_test.mean()        
    print('\nSVM Train with lr = {} & beta = {}'.format(lr, beta))
    print('Train accuracy = {}'.format(accuracy_train))
    print('Test accuracy = {}'.format(accuracy_test))
    print('Train hinge loss = {}'.format(hinge_train))             
    print('Test hinge loss = {}'.format(hinge_test)) 
    
    
    #Plot the weights
    w = svm_object.w[1:].reshape(28,28)
    plt.imshow(w, cmap='gray')
    plt.show()              
    


if __name__ == '__main__':
    
    #For beta 0.0
    sgd = GDOptimizer(1,0.0)
    w_history_0 = optimize_test_function(sgd)
    print("For beta 0.0: \n{}\nLength of w history: {}\n".format(w_history_0, len(w_history_0)))
    
    #For beta 0.9
    sgd = GDOptimizer(1,0.9)
    w_history_9 = optimize_test_function(sgd)
    print("For beta 0.9: \n{}\nLength of w history: {}\n".format(w_history_9, len(w_history_9)))
    
    #Plot the weights
    x = np.arange (0,len(w_history_0))
    plt.plot(x, w_history_0)
    plt.plot(x, w_history_9)    
    plt.legend(['Beta = 0.0', 'Beta = 0.9'], loc='upper right')
    plt.show()         


    #SVM Training 
    #intialize
    train_data, train_targets, test_data, test_targets=load_data() #load the data
    train_data = np.concatenate((np.ones((train_data.shape[0],1)),train_data),axis=1)  #Add Bias to data 
    test_data = np.concatenate((np.ones((test_data.shape[0],1)),test_data),axis=1)  #Add Bias to data 
    
    #Train svm & get results for beta = 0.0
    train_svm(0.05, 0.0, train_data, train_targets, test_data, test_targets)
    
    #Train svm & get results for beta = 0.1
    train_svm(0.05, 0.1, train_data, train_targets, test_data, test_targets)