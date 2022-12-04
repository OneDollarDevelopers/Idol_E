from facenet_pytorch import MTCNN, InceptionResnetV1
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

embeddings_list, labels = load_pkl()



print(len(embeddings_list))
print(len(labels))
print(type(embeddings_list[0]))
#list to numpy array

    
#from embeddings_list and labels split train and test data
def split_train_test():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(embeddings_list, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_train_test()



X_train = torch.stack(X_train, dim=0)
X_test = torch.stack(X_test, dim=0)
X_train = X_train.reshape(X_train.shape[0], 512)
X_test = X_test.reshape(X_test.shape[0], 512)
X_train = X_train.detach().numpy()
X_test = X_test.detach().numpy()
y_train = np.array(y_train)
y_test = np.array(y_test)

#train model SVM
def train_model():
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    return clf

clf = train_model()

#save model
def save_model():
    import pickle
    with open('model.pkl', 'wb') as f:
        print("done")
        pickle.dump(clf, f)


save_model()