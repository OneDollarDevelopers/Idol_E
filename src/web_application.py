import wget
import video_making as vm
from flask import Flask, render_template, request, jsonify, send_file
from flask_ngrok import run_with_ngrok
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import time
from datetime import timedelta

app = Flask(__name__)
run_with_ngrok(app)
@app.route('/')
def main():
    return render_template("main.html")
@app.route('/videocheck', methods=['POST', 'GET'])
def videocheck():
    if request.method == 'POST':
        
        input_video = request.files['input_video']
        input_video.save('../test_video/' + secure_filename(input_video.filename))
        path = Path('../test_video/' + secure_filename(input_video.filename))
        path.rename('../test_video/input.mp4')
        files = os.listdir('../test_video')
        return render_template('videocheck.html')
@app.route('/result', methods=['POST', 'GET'])
def make():
    if request.method == 'POST':
        groupname = request.form['input_group']
        member = request.form['input_member']
        start = time.process_time()
        final_clip = vm.find_clip('../test_video/input.mp4', groupname, member)
        final_clip.write_videofile("../test_video/result.mp4",  codec='libx264', 
                     audio_codec='aac', 
                     temp_audiofile='temp-audio.m4a', 
                     remove_temp=True)
        end = time.process_time()
        print("Time elapsed: ", timedelta(seconds=end-start))
        return render_template("result.html")
@app.route('/finish', methods=['POST', 'GET'])
def download():
    if request.method =='POST':
        return send_file('../test_video/result.mp4', as_attachment=True)

        
if __name__== "__main__":
    app.run()