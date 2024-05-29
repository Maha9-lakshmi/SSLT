# Speech to Sign Language Translation(SSLT)
The SSLT takes speech as input and creates a video of sign motions based on speech. We utilize the KNN algorithm to forecast the video.If the specified speech is not found in the database, the text is broken into individual letters or words and searched for the video corresponding to them using the KNN model. Later, the individual movies are combined into a single video of sign gestures.A small web application was created for the user interface to record audio from a microphone and capture video of sign movements.
The SSLT contains:
1. Speech Recognizer
2. Speech to text Translator
3. KNN model
4. MoivePy
5. HTML
6. Flask
