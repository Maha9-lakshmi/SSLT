from flask import Flask, render_template, request, send_file,url_for
from sklearn.model_selection import train_test_split
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from moviepy.editor import concatenate_videoclips, VideoFileClip
import cv2
import speech_recognition as sr

app = Flask(__name__)

# Step 1: Data Preparation
database_folder = "ISL_Gifs"
video_files = os.listdir(database_folder)
video_names = [os.path.splitext(video)[0] for video in video_files]

# Step 2: Feature Extraction and Model Training
vectorizer = TfidfVectorizer()
X_train, X_test, y_train, y_test = train_test_split(video_names, video_files, test_size=0.2, random_state=42)
X_train_transformed = vectorizer.fit_transform(X_train)
n_neighbors = 56   # Adjust the number of neighbors
metric = 'euclidean'
knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
knn_model.fit(X_train_transformed)

# Function to predict video based on text input
def predict_video(text_input):
    text_vector = vectorizer.transform([text_input])
    _, indices = knn_model.kneighbors(text_vector)
    predicted_video_index = indices[0][0]
    predicted_video = video_files[predicted_video_index]
    return predicted_video

# Step 3: Predict on Test Data 
correct_predictions = 0
total_predictions = len(X_test)

for text_input, expected_video in zip(X_test, y_test):
    predicted_video = predict_video(text_input)
    if predicted_video == expected_video:
        correct_predictions += 1

# create video for individual letters and merge them
def merge_letter_videos(letters, resize_width=640):
    letter_clips = []
    for letter in letters:
        letter_video_path = os.path.join(database_folder, f"{letter}.mp4")
        if os.path.exists(letter_video_path):
            letter_clip = VideoFileClip(letter_video_path)
            # Resize letter clip to maintain aspect ratio
            height = int(letter_clip.h * resize_width / letter_clip.w)
            letter_clip = letter_clip.resize(width=resize_width, height=height)
            letter_clips.append(letter_clip)
        else:
            print(f"Video for letter '{letter}' not found.")

    if letter_clips:
        return concatenate_videoclips(letter_clips)
    else:
        return None

# merge and display videos using predicted words from the model
def lmerge_and_display_videos(text):
    predicted_clips = []
    remaining_text = text

    # Iterate over the text to predict and merge words
    while remaining_text:
        found_word = False
        for i in range(len(remaining_text), 0, -1):
            word_to_predict = remaining_text[:i]
            predicted_video = predict_video(word_to_predict)
            if predicted_video and os.path.exists(os.path.join(database_folder, f"{predicted_video}.mp4")):
                predicted_clips.append(VideoFileClip(os.path.join(database_folder, f"{predicted_video}.mp4")))
                remaining_text = remaining_text[i:]
                found_word = True
                break
        if not found_word:
            # If no word is found, merge individual letters
            letter = remaining_text[0]
            letter_clip = merge_letter_videos([letter])
            if letter_clip:
                predicted_clips.append(letter_clip)
                remaining_text = remaining_text[1:]

    # Merge all predicted clips
    if predicted_clips:
        merged_clip = concatenate_videoclips(predicted_clips)
        # Resize the merged clip
        merged_clip = merged_clip.resize(width=640)
        video_path= "static/merged_video.mp4"
        merged_clip.write_videofile(video_path, codec="libx264", bitrate="5000k")
        print("Merged video saved as merged_video.mp4")
    else:
        print("No valid words found in the database.")

# Function to merge and display videos using predicted words from the model
def smerge_and_display_videos(text):
    predicted_clips = []
    for word in text.split():
        if os.path.exists(os.path.join(database_folder, f"{word}.mp4")):
            predicted_clips.append(VideoFileClip(os.path.join(database_folder, f"{word}.mp4")))
        else:
            print(f"Video for word '{word}' not found.")
            word_clip = merge_letter_videos(list(word))
            if word_clip:
                predicted_clips.append(word_clip)
    
    # Merge all predicted clips
    if predicted_clips:
        merged_clip = concatenate_videoclips(predicted_clips)
        # Resize the merged clip
        
        merged_clip = merged_clip.resize(width=640)
        video_path= "static/merged_video.mp4"
        merged_clip.write_videofile(video_path, codec="libx264", bitrate="5000k")
        print("Merged video saved as merged_video.mp4")
        display_video("merged_video.mp4")
    else:
        print("No valid words found in the database.")

# Alternative display_video function using OpenCV
# Alternative display_video function using OpenCV
def display_video(video_file):
    return video_file  

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video', methods=['POST'])
def process_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print('You Said: ' + text)
        # Call the function to merge and display videos
        smerge_and_display_videos(text)
        video_url = url_for('static', filename='merged_video.mp4')
        return render_template('video.html',video_url=video_url)
    except sr.UnknownValueError:
        print("could not understand audio")
        return "could not understand audio"
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service:", e)
        return "Could not request results from Google Speech Recognition service:", e

if __name__ == "__main__":
    #  Evaluate Accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print("Accuracy:", accuracy)
    app.run(debug=True)