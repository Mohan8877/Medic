import os
import sys
import re
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, join_room, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import uuid
import logging
from gemini_service import get_self_care_advice, get_fallback_diagnosis # Update import
# --- IMPORT GEMINI SERVICE ---
try:
    from gemini_service import get_self_care_advice
except ImportError as e:
    logging.error(f"❌ Failed to import gemini_service: {e}")
    # Fallback function if the file is missing
    def get_self_care_advice(disease):
        return "Error: Gemini service module not found."

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

# --- IMPORT ML MODEL ---
try:
    from models.inference.predictor import ModelPredictor
except ImportError as e:
    logging.error(f"❌ Failed to import ModelPredictor: {e}. Check your folder structure.")

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- INITIALIZE ML PREDICTOR ---
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'health_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'feature_names.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'label_encoder.pkl')

try:
    predictor = ModelPredictor(MODEL_PATH, FEATURES_PATH, ENCODER_PATH)
    logging.info("✅ ML Engine successfully hooked into Flask!")
except Exception as e:
    logging.warning(f"❌ ML Model not found. Did you run the training scripts? Error: {e}")
    predictor = None

# --- SYMPTOM PARSER HELPER ---
def extract_symptoms_from_text(text, feature_names):
    """Finds mentions of dataset features intelligently without over-activating."""
    text = text.lower()
    symptom_dict = {feature: 0 for feature in feature_names}
    detected_symptoms = set()
    
    sorted_features = sorted(feature_names, key=len, reverse=True)
    
    for feature in sorted_features:
        readable_feature = feature.replace('_', ' ')
        
        if re.search(r'\b' + re.escape(readable_feature) + r'\b', text):
            symptom_dict[feature] = 1
            detected_symptoms.add(readable_feature.title())
            text = text.replace(readable_feature, '')
            
    common_symptoms = ['headache', 'fever', 'cough', 'fatigue', 'nausea', 'vomiting', 'dizziness']
    for symptom in common_symptoms:
        if re.search(r'\b' + re.escape(symptom) + r'\b', text):
            if symptom in symptom_dict:
                symptom_dict[symptom] = 1
                detected_symptoms.add(symptom.title())
            elif symptom == 'fever' and 'high_fever' in symptom_dict:
                symptom_dict['high_fever'] = 1
                detected_symptoms.add('High Fever')

    if 'chest' in text and 'pain' in text and 'chest_pain' in symptom_dict:
        symptom_dict['chest_pain'] = 1
        detected_symptoms.add('Chest Pain')
        
    if 'breath' in text and 'breathlessness' in symptom_dict:
        symptom_dict['breathlessness'] = 1
        detected_symptoms.add('Breathlessness')

    detected_list = list(detected_symptoms)
    print(f"\n🧠 [AI PARSER] Symptoms Activated: {detected_list}\n")
            
    return symptom_dict, detected_list

# --- DATABASE MODELS ---
class Consultation(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    patient_name = db.Column(db.String(100))
    doctor_id = db.Column(db.String(50))
    status = db.Column(db.String(20), default="waiting")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.String(50), nullable=False)
    sender = db.Column(db.String(20), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('landingpage.html')
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')
@app.route('/login')
def login_page():
    return render_template('login.html')

# === ML PREDICTION API ===
from gemini_service import get_self_care_advice, get_fallback_diagnosis # Update import

@app.route('/api/analyze', methods=['POST'])
def analyze_symptoms():
    data = request.json
    user_text = data.get('text', '')
    symptom_dict, detected_list = extract_symptoms_from_text(user_text, predictor.feature_names)
    
    if len(detected_list) < 1:
        return jsonify({"error": "No symptoms detected."}), 400
        
    # 1. Try Local ML Model
    result = predictor.predict(symptom_dict)
    disease = str(result.get('prediction', 'Unknown'))
    confidence = int(result.get('confidence_score', 0.5) * 100)
    
    level = 'moderate' # default
    
    # 2. IF LOCAL MODEL IS UNSURE, ASK GEMINI
    if confidence < 35:
        print("Local model unsure. Consulting Gemini...")
        gemini_result = get_fallback_diagnosis(user_text)
        disease = gemini_result.get('diagnosis', 'Inconclusive')
        confidence = gemini_result.get('score', 40)
        level = gemini_result.get('level', 'moderate').lower()
    else:
        # Determine level for local model
        level = 'critical' if any(w in user_text.lower() for w in ['breath', 'chest']) else 'high' if confidence > 75 else 'moderate'

    return jsonify({
        "diagnosis": disease,
        "score": confidence,
        "level": level,
        "symptoms": detected_list
    })
# === NEW: GEMINI SELF-CARE API ===
@app.route('/api/selfcare', methods=['POST'])
def generate_selfcare():
    data = request.json
    disease = data.get('disease')
    
    if not disease:
        return jsonify({"error": "No disease provided"}), 400
        
    # Call your Gemini function
    advice_markdown = get_self_care_advice(disease)
    
    return jsonify({"advice": advice_markdown})

# --- CHAT & CONSULTATION ROUTES ---
@app.route('/api/start_consultation', methods=['POST'])
def start_consultation():
    data = request.json if request.is_json else request.form
    patient_name = data.get('patient_name', 'Anonymous')
    doctor_id = data.get('doctor_id')
    room_id = str(uuid.uuid4())[:8] 
    
    new_consult = Consultation(id=room_id, patient_name=patient_name, doctor_id=doctor_id)
    db.session.add(new_consult)
    db.session.commit()
    
    socketio.emit(f'new_patient_for_{doctor_id}', {'room_id': room_id, 'patient_name': patient_name})
    
    if request.is_json: 
        return jsonify({"chat_url": f"/patient/{room_id}"})
    return redirect(url_for('patient_chat', room_id=room_id))

@app.route('/patient/<room_id>')
def patient_chat(room_id): 
    return render_template('patient.html', room_id=room_id, role='patient')

@app.route('/doctor/dashboard')
def doctor_dashboard(): 
    return render_template('doctor_dashboard.html', patients=Consultation.query.filter_by(status='waiting').all())

@app.route('/doctor/<room_id>')
def doctor_chat(room_id): 
    return render_template('doctor.html', room_id=room_id, role='doctor')

# --- WEBSOCKET LOGIC ---
@socketio.on('join')
def on_join(data): 
    join_room(data['room'])

@socketio.on('send_message')
def handle_message(data):
    new_msg = Message(room_id=data['room'], sender=data['sender'], text=data['text'])
    db.session.add(new_msg)
    db.session.commit()
    emit('receive_message', {'text': data['text'], 'sender': data['sender']}, room=data['room'])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
