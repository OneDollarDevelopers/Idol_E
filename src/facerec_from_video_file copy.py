
import cv2
from moviepy.editor import *
import tqdm
import predict_module as M

input_movie = cv2.VideoCapture("../test_video/loona.mp4")
parent_clip = VideoFileClip('../test_video/loona.mp4', audio=True)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

print("please input name:")
favorite_name = input()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
videoclip=[]
audioclip=[]
frame_number = 0
start_time = 0
finish_time = 0
second = round(input_movie.get(cv2.CAP_PROP_FPS))
print(length)
flag= False
target_frame = []

#read frame per 15 frame recognize face and append to videoclip
for i in tqdm.tqdm(range(0, length, 15)):
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = input_movie.read()
    if not ret:
        print("break")
        break
    rgb_frame = frame[:, :, ::-1]
    face_names, face_encodings = M.predict(frame, groupname='Loona')
    if face_names == None:
        continue
    if favorite_name in face_names:
        target_frame.append(i)
    
#make videoclip
for i in tqdm.tqdm(range(len(target_frame))):
    if i == 0:
        start_time = target_frame[i]//second
        finish_time = start_time + 1
        videoclip.append(
            parent_clip.subclip(start_time, finish_time)
        )
    else:
        if target_frame[i] - target_frame[i-1] > 30:
            start_time = target_frame[i]//second
            finish_time = start_time + 1
            videoclip.append(
                parent_clip.subclip(start_time, finish_time)
            )       
        

# All done!
final_clip = concatenate_videoclips(videoclip)
final_clip.write_videofile("../test_video/moviepy1.mp4",  codec='libx264', 
                     audio_codec='aac', 
                     temp_audiofile='temp-audio.m4a', 
                     remove_temp=True)

input_movie.release()
cv2.destroyAllWindows()
