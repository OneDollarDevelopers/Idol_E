

#load model
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import PIL
import re
import tqdm


def load_model():
    import pickle
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

#predict
def predict():
    path = 'test/jisoo.jpg'
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20) #keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    clf = load_model()
    img = PIL.Image.open(path)
    img_cropped = mtcnn(img, save_path=None)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    img_embedding = img_embedding.detach().numpy()
    prediction = clf.predict(img_embedding)
    print(path)
    print(clf.predict(img_embedding))

predict()
print('Done')