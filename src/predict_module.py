#load model
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import PIL
import re
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
#is_hidden function
def is_hidden(path):
    return bool(re.search(r'^\.', path))


def load_model(groupname):
    import pickle
    with open('../'+groupname+'_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf



def predict(image, groupname):
    transform = T.ToPILImage()
    GN = groupname
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, keep_all=True) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    clf = load_model(GN)
    img_cropped = mtcnn(image, save_path=None)
    if type(img_cropped) == type(None):
        return None, None
    y_pred = []
    #print(str(len(img_cropped))+" faces found")
    for i in range(len(img_cropped)):
        #img = transform(img_cropped[i])
        #img.show()
        if type(img_cropped) == type(None):
            continue
        img_embedding = resnet(img_cropped[i].unsqueeze(0))
        img_embedding = img_embedding.detach().numpy()
       # print(np.amax(clf.predict_proba(img_embedding)))
        if (np.amax(clf.predict_proba(img_embedding)) > 0.8).astype(bool):
            y_pred.append(clf.predict(img_embedding))
            #print(y_pred[i])
        else:
            y_pred.append( 'unknown')
            #print(y_pred[i])
    return y_pred, img_embedding

