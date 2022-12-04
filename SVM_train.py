from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import PIL
import re
import tqdm
import gc

#load pkls
def load_pkls():
    import pickle
    with open('./embeddings_list.pkl', 'rb') as f:
        embeddings_list = pickle.load(f)
    with open('./labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    return embeddings_list, labels

embeddings_list, labels = load_pkls()


print(len(embeddings_list))
print(len(labels))
#from embeddings_list and labels split train and test data
def split_train_test():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(embeddings_list, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test()

#train model
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
        pickle.dump(clf, f)


