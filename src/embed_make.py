from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import PIL
import re
import tqdm
import gc

def is_hidden(path):
    return bool(re.search(r'^\.', path))

train_path = './train/'
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




#make embeddings list from train forder
def make_embeddings_list():
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings_list = []
    for i in tqdm.tqdm(range(len(train_list))):
        img = PIL.Image.open(train_list[i])
        img_cropped = mtcnn(img, save_path=None)
        del(img)
        if type(img_cropped) == type(None):
            del(labels[i])
            continue
        with torch.no_grad():
            img_embedding = resnet(img_cropped.unsqueeze(0))
        embeddings_list.append(img_embedding)
    return embeddings_list

#embeddings_list to file
def save_embeddings_list():
    import pickle
    with open('./embeddings_list.pkl', 'wb') as f:
        pickle.dump(embeddings_list, f)

#labels to file
def save_labels():
    import pickle
    with open('./labels.pkl', 'wb') as f:
        pickle.dump(labels, f)


train_list, labels = load_data()
embeddings_list = make_embeddings_list()
save_embeddings_list()
save_labels()





