#load model
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import PIL
import re
import numpy as np
#is_hidden function
def is_hidden(path):
    return bool(re.search(r'^\.', path))


def load_model():
    import pickle
    with open('./model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

#predict multiple images
def predict_multiple():
    import os
    from PIL import Image
    import numpy as np
    test_list = []
    for i in os.listdir('./test'):
        if not is_hidden(i):
            test_list.append('./test/'+i)
    for i in test_list:
        predict(i)

def predict(path):
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=True) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    clf = load_model()
    img = PIL.Image.open(path)
    img_cropped = mtcnn(img, save_path=None)
    y_pred = []
    print(str(len(img_cropped))+" faces found")
    for i in range(len(img_cropped)):
        img_embedding = resnet(img_cropped[i].unsqueeze(0))
        img_embedding = img_embedding.detach().numpy()
        if (np.amax(clf.predict_proba(img_embedding)) > 0.3).astype(bool):
            y_pred.append(clf.predict(img_embedding))
        else:
            y_pred.append( 'unknown')
    print(y_pred)


#predict_multiple()
predict('./test_embed/article.jpg')
print('Done')
