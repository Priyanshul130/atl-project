import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import pyttsx3
import speech_recognition as sr
from deepface import DeepFace
from textblob import TextBlob
import google.generativeai as genai

# Initialize AI, speech libraries, and DeepFace
recognizer = sr.Recognizer()
engine = pyttsx3.init()
genai.configure(api_key='AIzaSyAq30syoszI8LLlH8UQYPJsWH9N5u7yv9w')
chat = genai.GenerativeModel('gemini-pro').start_chat(history=[])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize GUI
root = tk.Tk()
root.title("AI Assistant")

# Create notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create AI tab
ai_tab = ttk.Frame(notebook)
notebook.add(ai_tab, text="AI Interaction")

text_area = tk.Text(ai_tab, wrap=tk.WORD, height=20, width=60)
text_area.pack(padx=10, pady=10)

language_var = tk.StringVar(value='en')
language_menu = ttk.Combobox(ai_tab, textvariable=language_var, values=['en', 'es', 'fr', 'de', 'hi'], state='readonly')
language_menu.pack(padx=10, pady=5)

start_button = tk.Button(ai_tab, text="Start AI", command=lambda: start_ai())
start_button.pack(padx=10, pady=5)

# Create Video tab
video_tab = ttk.Frame(notebook)
notebook.add(video_tab, text="Live Video")

video_label = tk.Label(video_tab)
video_label.pack(padx=10, pady=10)

def speak_text(text):
    language = language_var.get()
    engine.setProperty('voice', get_voice_id_for_language(language))
    engine.say(text)
    engine.runAndWait()

def get_voice_id_for_language(language_code):
    voices = engine.getProperty('voices')
    for voice in voices:
        if language_code in voice.languages:
            return voice.id
    return voices[0].id  # Default to first available voice

def estimate_stress(emotion):
    stress_emotions = ['angry', 'sad']
    if emotion in stress_emotions:
        return "High"
    else:
        return "Low"

def analyze_voice_stress(audio_data):
    # Placeholder function for analyzing voice stress
    try:
        pitch = np.random.rand()  # Replace with real pitch extraction
        if pitch > 0.7:
            return "High"
        else:
            return "Low"
    except Exception as e:
        print(f"Error analyzing voice stress: {e}")
        return "Unknown"

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

def capture_video():
    global video_source
    video_source = cv2.VideoCapture(0)
    if not video_source.isOpened():
        print("Could not open video device")
        return

    while True:
        ret, frame = video_source.read()
        if not ret:
            print("Failed to capture video frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            stress_level = estimate_stress(emotion)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"{emotion} - Stress: {stress_level}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video_label.config(image=photo)
        video_label.image = photo
        root.update_idletasks()

    video_source.release()

def start_video_thread():
    video_thread = threading.Thread(target=capture_video, daemon=True)
    video_thread.start()

def process_speech():
    language = language_var.get()
    with sr.Microphone() as source:
        try:
            audio_data = recognizer.listen(source)
            text = recognizer.recognize_google(audio_data, language=language)
            text_area.insert(tk.END, f"YOU SAID = {text}\n")

            sentiment = analyze_sentiment(text)
            voice_stress = analyze_voice_stress(audio_data)
            text_area.insert(tk.END, f"Sentiment: {sentiment}\n")
            text_area.insert(tk.END, f"Voice Stress: {voice_stress}\n")
            
            handle_voice_command(text)
        except (sr.UnknownValueError, sr.RequestError) as e:
            text_area.insert(tk.END, f"Error: {str(e)}\n")
        root.after(1000, process_speech)

def handle_voice_command(text):
    commands = {
        'capture video': start_video_thread,
        'pause video': lambda: None,
        'resume video': lambda: None,
        'screenshot': lambda: None,
        'show weather': lambda: None,
        'show news': lambda: None,
        'show emotion plot': lambda: None,
        'stress management tips': lambda: None,
        'study tips': lambda: None,
        'parental support': lambda: None,
        'mental health awareness': lambda: None,
        'interactive sessions': lambda: None
    }
    
    action = commands.get(text.lower(), None)
    if action:
        action()
    else:
        response = chat.send_message(text)
        text_area.insert(tk.END, f"AI: {response.text}\n")
        speak_text(response.text)

def start_ai():
    speak_text('Welcome to AI WORLD! MY NAME IS VIRAT KOHLI')
    text_area.insert(tk.END, 'Welcome to AI WORLD! MY NAME IS VIRAT KOHLI\n')
    text_area.insert(tk.END, "Please ask some questions, I like to talk to you.\n")
    process_speech()
def on_video_tab_selected(event):
    if notebook.index(notebook.select()) == 1:
        start_video_thread()

notebook.bind("<<NotebookTabChanged>>", on_video_tab_selected)

# Start the Tkinter main loop
root.mainloop()

