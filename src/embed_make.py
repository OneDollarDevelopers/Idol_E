from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import PIL
import re
import tqdm

def is_hidden(path):
    return bool(re.search(r'^\.', path))

train_path = 'train_less images/'
#load image data from train folder with labels
def load_data():
    import os
    from PIL import Image
    import numpy as np
    train_list = []
    labels = []
    for i in os.listdir(train_path):
        if not is_hidden(i):
            for j in os.listdir(train_path+i):
                if not is_hidden(j):
                    train_list.append(train_path+i+'/'+j)
                    labels.append(i)
    return train_list, labels



train_list, labels = load_data()

#make embeddings list from train forder
def make_embeddings_list():
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings_list = []
    for i in tqdm.tqdm(range(len(train_list))):
        print(train_list[i])
        img = PIL.Image.open(train_list[i])
        img_cropped = mtcnn(img, save_path=None)
        img_embedding = resnet(img_cropped.unsqueeze(0))
        embeddings_list.append(img_embedding)
    return embeddings_list

embeddings_list = make_embeddings_list()

#SVM classifier for embeddings
def svm_classifier():
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings_list, labels)
    return clf

clf = svm_classifier()

#save model
def save_model():
    import pickle
    with open('./model.pkl', 'wb') as f:
        pickle.dump(clf, f)

save_model()

#load model
def load_model():
    import pickle
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

#predict
def predict():
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    clf = load_model()
    img = PIL.Image.open('test/1.jpg')
    img_cropped = mtcnn(img, save_path=None)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    print(clf.predict(img_embedding))

predict()
print('Done')



