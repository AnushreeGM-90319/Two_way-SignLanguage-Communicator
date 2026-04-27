# AI Two-Way Sign Language Communicator

A real-time communication system that bridges the gap between Sign Language and Spoken Language using AI.

---

FEATURES

* Camera Mode → Detects sign language using webcam
* Voice Mode → Converts speech to text
* Text-to-Speech → Speaks detected sign words
* Chat-style interface
* Real-time AI predictions

---

TECH STACK

* Frontend: HTML, CSS, JavaScript
* Backend: FastAPI
* AI Model: PyTorch (CNN + LSTM)
* Computer Vision: OpenCV, MediaPipe

---

HOW TO CLONE AND RUN

1. Clone the Repository

git clone https://github.com/AnushreeGM-90319/Two_way-SignLanguage-Communicator.git
cd Two_way-SignLanguage-Communicator

---

2. Create Virtual Environment

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

---

3. Install Dependencies

pip install -r requirements.txt

---

4. Run Backend Server

cd backend
uvicorn main_communicator:app --reload

---

5. Run Frontend

Open this file in your browser:

frontend/index2.html

(You can also use VS Code Live Server)

---

HOW TO USE

1. Open the app
2. Choose a mode:

   * Camera Mode → Sign detection
   * Voice Mode → Speech input
3. Start communicating

---

PROJECT STRUCTURE

Two_way-SignLanguage-Communicator/

backend/
main_communicator.py
main.py

frontend/
index.html
index2.html

requirements.txt
README.md

---

NOTES

* Enable camera and microphone permissions
* Run backend before frontend
* Keep sign_model.pth inside backend folder

---

AUTHOR

Anushree G M
