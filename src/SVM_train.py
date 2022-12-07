
import numpy as np
import torch


#load pkls to numpy arrays
def load_pkl():
    import pickle
    with open('./embeddings_list.pkl', 'rb') as f:  
        embeddings_list = pickle.load(f)
    with open('./labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    return embeddings_list, labels  



    
#from embeddings_list and labels split train and test data
def split_train_test():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(embeddings_list, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test




#train model SVM
def train_model():
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    return clf

#test model
def test_model():
    from sklearn.metrics import accuracy_score
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

#save model
def save_model():
    import pickle
    with open('model.pkl', 'wb') as f:
        print("done")
        pickle.dump(clf, f)

embeddings_list, labels = load_pkl()

X_train, X_test, y_train, y_test = split_train_test()



X_train = torch.stack(X_train, dim=0)
X_test = torch.stack(X_test, dim=0)
X_train = X_train.reshape(X_train.shape[0], 25)
X_test = X_test.reshape(X_test.shape[0], 25)
X_train = X_train.detach().numpy()
X_test = X_test.detach().numpy()
y_train = np.array(y_train)
y_test = np.array(y_test)

clf = train_model()
test_model()
save_model()