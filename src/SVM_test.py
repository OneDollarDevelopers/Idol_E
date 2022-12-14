#load model
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import PIL
import re
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
#is_hidden function
def is_hidden(path):
    return bool(re.search(r'^\.', path))


def load_model():
    import pickle
    with open('../model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

#predict multiple images
def predict_multiple():
    import os
    from PIL import Image
    import numpy as np
    test_list = []
    for i in os.listdir('../test'):
        if not is_hidden(i):
            test_list.append('../test/'+i)
    for i in test_list:
        predict(i)

def detect_face(path):
    img = PIL.Image.open(path)
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, keep_all=True) #keep_all=True
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

    # Visualize
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.axis('off')
    
    for box, landmark in zip(boxes, landmarks):
        ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
        ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
        fig.show()

def predict(path):
    transform = T.ToPILImage()
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, keep_all=True) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    clf = load_model()
    img = PIL.Image.open(path)
    img_cropped = mtcnn(img, save_path=None)
    print(type(img_cropped[0]))
    y_pred = []
    print(str(len(img_cropped))+" faces found")
    for i in range(len(img_cropped)):
        #img = transform(img_cropped[i])
        #img.show()
        img_embedding = resnet(img_cropped[i].unsqueeze(0))
        img_embedding = img_embedding.detach().numpy()
        print(np.amax(clf.predict_proba(img_embedding)))
        if (np.amax(clf.predict_proba(img_embedding)) > 0.4).astype(bool):
            y_pred.append(clf.predict(img_embedding))
            print(y_pred[i])
        else:
            y_pred.append( ['unknown'])
            print(y_pred[i])
    return y_pred


#predict_multiple()
prediction = predict('../test_embed/article.jpg')
detect_face('../test_embed/article.jpg')
#print(prediction)
print('Done')
