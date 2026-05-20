import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from detection import SignDetector
import threading
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Detection System")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f4f4f9")
        
        # Core detector (MediaPipe + KMeans)
        self.detector = SignDetector()
        
        # States
        self.is_running = False
        self.cap = None
        self.history = []
        
        # Log file
        self.log_file = "predictions.log"
        
        self.setup_ui()
        
    def setup_ui(self):
        """
        Creates the Tkinter Layout: Camera View, Word Display, Logs, and Buttons
        """
        # Header Label
        header = tk.Label(self.root, text="Sign Language Translator", font=("Helvetica", 24, "bold"), bg="#f4f4f9", fg="#333")
        header.pack(pady=10)
        
        # Video Container
        self.video_frame = tk.Frame(self.root, bg="black", width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        # Word Predictions
        self.word_var = tk.StringVar(value="Waiting for camera...")
        word_label = tk.Label(self.root, textvariable=self.word_var, font=("Helvetica", 40, "bold"), fg="#28a745", bg="#f4f4f9")
        word_label.pack()
        
        # Confidence Score
        self.conf_var = tk.StringVar(value="Confidence: --%")
        conf_label = tk.Label(self.root, textvariable=self.conf_var, font=("Helvetica", 14), bg="#f4f4f9", fg="#666")
        conf_label.pack(pady=5)
        
        # History & Sentence Formation
        self.history_var = tk.StringVar(value="Raw Words: ")
        history_label = tk.Label(self.root, textvariable=self.history_var, font=("Helvetica", 14, "italic"), bg="#f4f4f9", fg="#17a2b8")
        history_label.pack(pady=5)
        
        # AI Sentence Generation Feature
        ai_frame = tk.Frame(self.root, bg="#f4f4f9")
        ai_frame.pack(pady=5)
        
        tk.Label(ai_frame, text="Google Gemini API Key:", bg="#f4f4f9").grid(row=0, column=0, padx=5)
        self.api_key_var = tk.StringVar(value=os.getenv("GEMINI_API_KEY", ""))
        self.api_key_entry = ttk.Entry(ai_frame, textvariable=self.api_key_var, show="*", width=30)
        self.api_key_entry.grid(row=0, column=1, padx=5)
        
        self.btn_ai = ttk.Button(ai_frame, text="✨ Generate Real Sentence", command=self.generate_ai_sentence)
        self.btn_ai.grid(row=0, column=2, padx=5)
        
        self.ai_sentence_var = tk.StringVar(value="AI Sentence: ")
        ai_label = tk.Label(self.root, textvariable=self.ai_sentence_var, font=("Helvetica", 22, "bold"), bg="#f4f4f9", fg="#d9534f")
        ai_label.pack(pady=10)
        
        # Control Buttons
        controls = tk.Frame(self.root, bg="#f4f4f9")
        controls.pack(pady=10)
        
        self.btn_start = ttk.Button(controls, text="Start Detection", command=self.start_camera)
        self.btn_start.grid(row=0, column=0, padx=15, ipadx=10, ipady=5)
        
        self.btn_stop = ttk.Button(controls, text="Stop Detection", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=1, padx=15, ipadx=10, ipady=5)
        
        self.btn_tts = ttk.Button(controls, text="Speak Current Word", command=self.speak_word)
        self.btn_tts.grid(row=0, column=2, padx=15, ipadx=10, ipady=5)

    def speak_word(self):
        """ Runs Text-To-Speech conversion """
        word = self.word_var.get()
        if word and word not in ["Waiting for camera...", "Waiting..."]:
            threading.Thread(target=self.detector.text_to_speech, args=(word,)).start()

    def generate_ai_sentence(self):
        """ Sends the raw words to Gemini API to form a natural sentence """
        api_key = self.api_key_var.get().strip()
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            
        if not api_key:
            self.ai_sentence_var.set("AI Sentence: [Please configure .env or paste API Key]")
            return
            
        words = self.history
        if not words:
            self.ai_sentence_var.set("AI Sentence: [No words detected yet]")
            return
            
        self.ai_sentence_var.set("AI Sentence: Generating...")
        
        def run_ai():
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                if not valid_models:
                    self.root.after(0, lambda: self.ai_sentence_var.set("AI Sentence: [Error: No available model found for this key]"))
                    return
                    
                model_name = valid_models[0]
                for name in valid_models:
                    if 'flash' in name:
                        model_name = name
                        break
                        
                model = genai.GenerativeModel(model_name) 
                
                # Prompt instructing the model to translate raw words into a valid sentence
                prompt = (
                    "You are a sign language translator. "
                    "Convert the following sequence of sign language words into a proper, "
                    "grammatically correct, and natural English sentence. "
                    "Only reply with the sentence, nothing else.\n\n"
                    f"Words: {' '.join(words)}"
                )
                
                response = model.generate_content(prompt)
                sentence = response.text.strip()
                
                # Update UI thread-safely
                self.root.after(0, lambda: self.ai_sentence_var.set("AI Sentence: " + sentence))
                # Speak it out loud
                self.root.after(0, lambda: threading.Thread(target=self.detector.text_to_speech, args=(sentence,)).start())
                
            except ImportError:
                self.root.after(0, lambda: self.ai_sentence_var.set("AI Sentence: [Error: Please install google-generativeai]"))
            except Exception as e:
                self.root.after(0, lambda: self.ai_sentence_var.set(f"AI Sentence: [API Error: {str(e)[:40]}]"))
                print(f"AI Error: {e}")
                
        threading.Thread(target=run_ai).start()
            
    def log_prediction(self, word, confidence):
        """ Appends detection to a log file """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] Word: {word} | Confidence: {confidence:.2f}%\n")
            
    def update_history(self, word, confidence):
        """ Updates the sliding window forming sentences """
        # Don't add duplicate subsequent words easily
        if not self.history or self.history[-1] != word:
            self.history.append(word)
            if len(self.history) > 10:
                self.history.pop(0) # Keep max 10 words history
                
            self.history_var.set("Raw Words: " + " ".join(self.history))
            
            # Speak real-time detections out loud automatically
            self.speak_word()
            # Log it for final year demo requirements
            self.log_prediction(word, confidence)

    def start_camera(self):
        """ Opens cv2.VideoCapture """
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.run_detector_loop()
        
    def stop_camera(self):
        """ Turns off the camera stream """
        self.is_running = False
        if self.cap:
            self.cap.release()
            
        self.video_label.config(image='')
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.word_var.set("Waiting for camera...")
        self.conf_var.set("Confidence: --%")
        
    def run_detector_loop(self):
        """ Fetches frames smoothly using after() """
        if self.is_running and self.cap.isOpened():
            # Grabbing Frame
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame naturally
                frame = cv2.flip(frame, 1)
                
                # Analyze using detector class
                processed_frame, detected_word, confidence = self.detector.detect_signs_real_time(frame)
                
                # Update UI elements
                if confidence > 80.0:
                    self.word_var.set(detected_word)
                    self.conf_var.set(f"Confidence: {confidence:.1f}%")
                    self.update_history(detected_word, confidence)
                
                # Push OpenCV Matrix back to Tkinter Pillow Image
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                # Prevent GC sweep via reference matching
                self.video_label.img_tk = img_tk
                self.video_label.configure(image=img_tk)
                
            # Schedule next frame approx 30 FPS (~30ms)
            self.root.after(30, self.run_detector_loop)

if __name__ == "__main__":
    # Test boot GUI
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
