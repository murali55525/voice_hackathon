import face_recognition
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import sqlite3
from cryptography.fernet import Fernet
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import pickle
import logging
from pydub import AudioSegment
import tempfile
import librosa
import mediapipe as mp
import shutil
from datetime import datetime
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'biometric_data')
USERS_DIR = os.path.join(DATA_DIR, 'users')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Set up database
def migrate_database(conn):
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (user_id INTEGER PRIMARY KEY, face_model BLOB, voice_model BLOB, 
                      face_encoding BLOB, voice_encoding BLOB, face_image BLOB,
                      voice_audio BLOB)''')
        c.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in c.fetchall()]
        logger.debug(f"Current users table columns: {columns}")
        for col in ['face_model', 'voice_model', 'face_encoding', 'voice_encoding', 
                   'face_image', 'voice_audio']:
            if col not in columns:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} BLOB")
                logger.debug(f"Added {col} column")
        conn.commit()
        logger.debug("Database schema migration completed")
    except Exception as e:
        logger.error(f"Failed to migrate database schema: {str(e)}")
        raise

# Initialize database connection
try:
    conn = sqlite3.connect('biometric.db', check_same_thread=False)
    migrate_database(conn)
    logger.debug("Database connection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise

# Load deepfake detection models with error handling
try:
    face_deepfake_model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    face_processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    audio_deepfake_model = Wav2Vec2ForSequenceClassification.from_pretrained("mo-thecreator/Deepfake-audio-detection")
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("mo-thecreator/Deepfake-audio-detection")
    logger.debug("Deepfake detection models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load deepfake detection models: {str(e)}")
    raise

# Load speaker recognition model
try:
    voice_encoder = VoiceEncoder("cpu")
    logger.debug("Voice encoder loaded successfully")
except Exception as e:
    logger.error(f"Failed to load voice encoder: {str(e)}")
    raise

# Simple neural network for user-specific biometric authentication


class BiometricClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super(BiometricClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_same = label * torch.pow(euclidean_distance, 2)
        loss_diff = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_same + loss_diff)

def save_model(model):
    """Serialize a PyTorch model to bytes"""
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return None

def load_model(state_dict_bytes, input_dim):
    """Deserialize a PyTorch model from bytes"""
    try:
        model = BiometricClassifier(input_dim)
        buffer = io.BytesIO(state_dict_bytes)
        model.load_state_dict(torch.load(buffer))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Helper functions
def process_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        logger.debug(f"Processing face image base64 (first 50 chars): {base64_string[:50]}")
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Ensure RGB
        image_np = np.array(image, dtype=np.uint8)  # Explicitly set to uint8
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            logger.error("Image is not RGB (3 channels)")
            return None
        if image_np.dtype != np.uint8:
            logger.error(f"Image is not 8-bit (dtype: {image_np.dtype})")
            return None
        logger.debug(f"Image processed successfully, shape: {image_np.shape}, dtype: {image_np.dtype}")
        return image_np
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def validate_audio_file(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.duration_seconds < 1.0:
            logger.error("Audio file too short")
            return False
        if audio.frame_rate not in [16000, 44100, 48000]:
            logger.error(f"Unsupported sample rate: {audio.frame_rate}")
            return False
        logger.debug("Audio file validated successfully")
        return True
    except Exception as e:
        logger.error(f"Invalid audio file: {str(e)}")
        return False

def process_audio_file(audio_data):
    try:
        # Save temporary WebM file
        temp_webm = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
        with open(temp_webm, "wb") as f:
            f.write(audio_data)
        logger.debug(f"Audio file saved as {temp_webm}")

        # Convert WebM to WAV
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        audio = AudioSegment.from_file(temp_webm, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure 16kHz mono for Wav2Vec2
        audio.export(temp_wav, format="wav")
        logger.debug(f"Audio converted to {temp_wav}")

        # Validate the converted WAV file
        if not validate_audio_file(temp_wav):
            logger.error("Audio validation failed")
            os.remove(temp_webm)
            os.remove(temp_wav)
            return None

        # Clean up temporary WebM
        os.remove(temp_webm)
        return temp_wav
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

def check_face_liveness(image):
    """Enhanced face liveness detection with robust landmark handling"""
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False, "No face landmarks detected"
        
        landmarks = results.multi_face_landmarks[0].landmark

        # Define landmark indices for facial features
        FACIAL_LANDMARKS = {
            'nose_tip': 1,
            'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'left_ear': 234,
            'right_ear': 454,
            'mouth': [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
        }

        # Validate landmark indices
        for feature, indices in FACIAL_LANDMARKS.items():
            if isinstance(indices, list):
                if not all(0 <= idx < len(landmarks) for idx in indices):
                    return False, f"Invalid landmark index for {feature}"
            else:
                if not 0 <= indices < len(landmarks):
                    return False, f"Invalid landmark index for {feature}"

        # 1. Depth check
        nose_depth = landmarks[FACIAL_LANDMARKS['nose_tip']].z
        left_ear_pos = landmarks[FACIAL_LANDMARKS['left_ear']]
        right_ear_pos = landmarks[FACIAL_LANDMARKS['right_ear']]
        ear_distance = abs(left_ear_pos.x - right_ear_pos.x)
        
        if ear_distance == 0:  # Avoid division by zero
            return False, "Invalid ear distance detected"
            
        depth_ratio = abs(nose_depth) / ear_distance

        # 2. Eye openness check
        def get_eye_height(eye_points):
            top_point = min(landmarks[i].y for i in eye_points)
            bottom_point = max(landmarks[i].y for i in eye_points)
            return abs(top_point - bottom_point)

        left_eye_height = get_eye_height(FACIAL_LANDMARKS['left_eye'])
        right_eye_height = get_eye_height(FACIAL_LANDMARKS['right_eye'])
        
        # 3. Face angle check
        face_rotation = abs(left_ear_pos.z - right_ear_pos.z)

        # Combine all checks with adjusted thresholds
        checks = {
            "depth": 0.1 < depth_ratio < 1.0,  # More permissive depth ratio
            "eyes_open": min(left_eye_height, right_eye_height) > 0.01,
            "face_angle": face_rotation < 0.5
        }

        failed_checks = [k for k, v in checks.items() if not v]
        
        # Log detailed measurements for debugging
        logger.debug(f"Liveness measurements - depth_ratio: {depth_ratio:.3f}, "
                    f"eye_heights: {left_eye_height:.3f}/{right_eye_height:.3f}, "
                    f"face_rotation: {face_rotation:.3f}")

        if failed_checks:
            return False, f"Liveness check failed: {', '.join(failed_checks)}"

        return True, "Liveness check passed"

    except IndexError as e:
        logger.error(f"Landmark index error: {str(e)}")
        return False, "Invalid facial landmark detection"
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}")
        return False, f"Liveness check failed: {str(e)}"

def save_user_data(user_id, face_image_np, voice_wav):
    """Save user data in organized directory structure"""
    try:
        user_dir = os.path.join(USERS_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        # Save face image
        face_path = os.path.join(user_dir, 'face.jpg')
        cv2.imwrite(face_path, face_image_np)
        
        # Save voice recording
        voice_path = os.path.join(user_dir, 'voice.wav')
        AudioSegment.from_wav(voice_wav).export(voice_path, format='wav')
        
        # Save metadata
        metadata = {
            'enrolled_at': datetime.now().isoformat(),
            'face_shape': face_image_np.shape,
            'last_verified': None
        }
        with open(os.path.join(user_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        return True
    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")
        return False

def train_biometric_model(features, input_dim, user_id, epochs=30):
    """Train a biometric model for user authentication"""
    try:
        model = BiometricClassifier(input_dim)
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert features to tensor and normalize
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        features_tensor = features_tensor / torch.norm(features_tensor)

        # Generate positive samples with data augmentation
        positive_samples = [features_tensor]
        for _ in range(5):
            noise = torch.randn_like(features_tensor) * 0.02
            augmented = features_tensor + noise
            augmented = augmented / torch.norm(augmented)
            positive_samples.append(augmented)
        positive_samples = torch.cat(positive_samples, dim=0)

        # Generate negative samples from random noise
        negative_samples = torch.randn(5, input_dim)
        negative_samples = negative_samples / torch.norm(negative_samples, dim=1, keepdim=True)

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            # Train on positive pairs
            for i in range(len(positive_samples)):
                for j in range(i + 1, len(positive_samples)):
                    optimizer.zero_grad()
                    out1 = model(positive_samples[i:i+1])
                    out2 = model(positive_samples[j:j+1])
                    loss = criterion(out1, out2, torch.tensor(1.0))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Train on negative pairs
            for pos in positive_samples:
                for neg in negative_samples:
                    optimizer.zero_grad()
                    out1 = model(pos.unsqueeze(0))
                    out2 = model(neg.unsqueeze(0))
                    loss = criterion(out1, out2, torch.tensor(0.0))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

        model.eval()
        return model

    except Exception as e:
        logger.error(f"Error in train_biometric_model: {str(e)}")
        return None

def enroll_user(user_id, face_image_np, voice_file):
    global conn
    voice_wav = None
    try:
        logger.debug("Attempting face detection")
        # Ensure image is uint8 RGB
        if face_image_np.dtype != np.uint8:
            face_image_np = face_image_np.astype(np.uint8)
        # Resize for face recognition
        face_image_np = cv2.resize(face_image_np, (128, 128))
        face_locations = face_recognition.face_locations(face_image_np, model="hog", number_of_times_to_upsample=2)
        if len(face_locations) == 0:
            logger.error("No face detected")
            return False, "No face detected"
        face_encoding = face_recognition.face_encodings(face_image_np, face_locations)[0]
        logger.debug("Face encoding generated")

        # Validate face encoding
        if face_encoding.shape != (128,):
            logger.error(f"Invalid face encoding shape: {face_encoding.shape}")
            return False, "Invalid face encoding"
        if np.any(np.isnan(face_encoding)) or np.any(np.isinf(face_encoding)):
            logger.error("Face encoding contains NaN or Inf values")
            return False, "Invalid face encoding"
        norm = np.linalg.norm(face_encoding)
        if norm == 0:
            logger.error("Face encoding has zero norm")
            return False, "Invalid face encoding"
        face_encoding = face_encoding / norm
        logger.debug(f"Face encoding validated: shape={face_encoding.shape}, norm={norm:.4f}")

        # Add liveness check
        is_live, liveness_msg = check_face_liveness(face_image_np)
        if not is_live:
            logger.error(f"Face liveness check failed: {liveness_msg}")
            return False, f"Face liveness check failed: {liveness_msg}"
        logger.debug("Face liveness check passed")

        logger.debug("Running face deepfake detection")
        inputs = face_processor(images=face_image_np, return_tensors="pt")
        with torch.no_grad():
            outputs = face_deepfake_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        real_prob = probs[:, 0].item()
        logger.info(f"Deepfake detection - Logits: {logits.tolist()}, Real prob: {real_prob:.4f}")
        is_real_face = real_prob > 0.3
        if not is_real_face:
            logger.error(f"Face flagged as deepfake (real prob: {real_prob:.4f})")
            return False, "Face is deepfake"
        logger.debug("Face passed deepfake check")

        try:
            # Create a copy of voice file for later use
            voice_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            AudioSegment.from_wav(voice_file).export(voice_wav, format='wav')
            
            logger.debug("Loading audio for deepfake detection")
            audio_data, sr = librosa.load(voice_file, sr=16000, mono=True)
            logger.debug(f"Audio loaded: shape={audio_data.shape}, sample_rate={sr}")
            audio_features = audio_feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                audio_outputs = audio_deepfake_model(**audio_features)
            audio_probs = torch.nn.functional.softmax(audio_outputs.logits, dim=-1)
            is_real_voice = audio_probs[:, 0].item() > 0.5
            if not is_real_voice:
                logger.error("Voice is deepfake")
                return False, "Voice is deepfake"
            logger.debug("Voice passed deepfake check")

            logger.debug("Generating voice encoding")
            wav = preprocess_wav(voice_file)
            voice_encoding = voice_encoder.embed_utterance(wav)
            # Validate voice encoding
            if np.any(np.isnan(voice_encoding)) or np.any(np.isinf(voice_encoding)):
                logger.error("Voice encoding contains NaN or Inf values")
                return False, "Invalid voice encoding"
            norm = np.linalg.norm(voice_encoding)
            if norm == 0:
                logger.error("Voice encoding has zero norm")
                return False, "Invalid voice encoding"
            voice_encoding = voice_encoding / norm
            logger.debug(f"Voice encoding validated: shape={voice_encoding.shape}, norm={norm:.4f}")
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return False, f"Audio processing failed: {str(e)}"
        finally:
            # Clean up original voice file
            if os.path.exists(voice_file):
                os.remove(voice_file)
                logger.debug("Original voice file removed")

        logger.debug("Training face model")
        face_model = train_biometric_model(face_encoding, input_dim=128, user_id=user_id)
        if face_model is None:
            logger.error("Failed to train face model")
            return False, "Failed to train face model"
        logger.debug("Training voice model")
        voice_model = train_biometric_model(voice_encoding, input_dim=voice_encoding.shape[0], user_id=user_id)
        if voice_model is None:
            logger.error("Failed to train voice model")
            return False, "Failed to train voice model"
        logger.debug("Biometric models trained")

        logger.debug("Saving face model")
        face_model_bytes = save_model(face_model)
        if face_model_bytes is None:
            logger.error("Failed to save face model")
            return False, "Failed to save face model"
        logger.debug("Saving voice model")
        voice_model_bytes = save_model(voice_model)
        if voice_model_bytes is None:
            logger.error("Failed to save voice model")
            return False, "Failed to save voice model"
        logger.debug("Model weights saved")

        # Save user data before database insertion
        if not save_user_data(user_id, face_image_np, voice_wav):
            return False, "Failed to save user data"
        logger.debug(f"User data saved to directory for user {user_id}")

        # Store original face image and voice data
        face_image_bytes = cv2.imencode('.jpg', face_image_np)[1].tobytes()
        voice_audio = AudioSegment.from_wav(voice_wav)
        voice_audio = voice_audio.set_frame_rate(16000).set_channels(1)
        voice_audio_buffer = io.BytesIO()
        voice_audio.export(voice_audio_buffer, format='wav', parameters=["-acodec", "pcm_s16le"])
        voice_audio_bytes = voice_audio_buffer.getvalue()

        logger.debug("Inserting user into database")
        c = conn.cursor()
        c.execute("""INSERT INTO users 
                    (user_id, face_model, voice_model, face_encoding, voice_encoding,
                     face_image, voice_audio) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, face_model_bytes, voice_model_bytes, 
                   pickle.dumps(face_encoding), pickle.dumps(voice_encoding),
                   face_image_bytes, voice_audio_bytes))
        conn.commit()
        logger.debug(f"User {user_id} enrolled in database")
        return True, "User enrolled successfully"
    except Exception as e:
        logger.error(f"Error in enroll_user: {str(e)}")
        return False, f"Enrollment failed: {str(e)}"
    finally:
        # Clean up voice_wav file
        if voice_wav and os.path.exists(voice_wav):
            try:
                os.remove(voice_wav)
                logger.debug("Temporary WAV file removed")
            except Exception as e:
                logger.warning(f"Failed to remove temporary WAV file: {str(e)}")

def authenticate_user(user_id, face_image_np, voice_file):
    """Authenticate user using face and voice biometrics"""
    stored_voice_file = None
    temp_voice_file = None
    try:
        # Validate face image
        face_locations = face_recognition.face_locations(face_image_np)
        if not face_locations:
            return False, "No face detected"
        
        # Generate face encoding
        face_encoding = face_recognition.face_encodings(face_image_np)[0]
        face_encoding = face_encoding / np.linalg.norm(face_encoding)

        # Verify liveness
        is_live, liveness_msg = check_face_liveness(face_image_np)
        if not is_live:
            return False, f"Face liveness check failed: {liveness_msg}"

        # Get stored user data
        c = conn.cursor()
        c.execute("""SELECT face_model, voice_model, face_encoding, voice_encoding 
                    FROM users WHERE user_id = ?""", (user_id,))
        result = c.fetchone()
        if not result:
            return False, "User not found"

        face_model_bytes, voice_model_bytes, stored_face_enc, stored_voice_enc = result
        stored_face_encoding = pickle.loads(stored_face_enc)
        stored_voice_encoding = pickle.loads(stored_voice_enc)

        # Direct face matching
        face_distance = face_recognition.face_distance([stored_face_encoding], face_encoding)[0]
        if face_distance > 0.6:
            return False, "Face does not match"

        # Load and verify voice
        try:
            wav = preprocess_wav(voice_file)
            voice_encoding = voice_encoder.embed_utterance(wav)
            voice_encoding = voice_encoding / np.linalg.norm(voice_encoding)
            
            # Direct voice matching
            voice_distance = np.linalg.norm(voice_encoding - stored_voice_encoding)
            if voice_distance > 0.6:
                return False, "Voice does not match"
        except Exception as e:
            return False, f"Voice verification failed: {str(e)}"

        # Load and apply biometric models
        face_model = load_model(face_model_bytes, input_dim=128)
        voice_model = load_model(voice_model_bytes, input_dim=voice_encoding.shape[0])
        
        if not face_model or not voice_model:
            return False, "Failed to load biometric models"

        # Model-based verification
        with torch.no_grad():
            face_input = torch.tensor(face_encoding, dtype=torch.float32).unsqueeze(0)
            voice_input = torch.tensor(voice_encoding, dtype=torch.float32).unsqueeze(0)
            
            face_emb = face_model(face_input)
            voice_emb = voice_model(voice_input)
            
            stored_face = torch.tensor(stored_face_encoding, dtype=torch.float32).unsqueeze(0)
            stored_voice = torch.tensor(stored_voice_encoding, dtype=torch.float32).unsqueeze(0)
            
            stored_face_emb = face_model(stored_face)
            stored_voice_emb = voice_model(stored_voice)
            
            face_match = torch.nn.functional.pairwise_distance(face_emb, stored_face_emb) < 0.3
            voice_match = torch.nn.functional.pairwise_distance(voice_emb, stored_voice_emb) < 0.3

        if face_match and voice_match:
            return True, "Authentication successful"
        else:
            return False, "Biometric verification failed"

    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return False, f"Authentication failed: {str(e)}"
    finally:
        # Clean up temporary files
        for temp_file in [voice_file, stored_voice_file, temp_voice_file]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {str(e)}")

# Add route to view stored user data
@app.route('/user/<int:user_id>/data', methods=['GET'])
def get_user_data(user_id):
    try:
        user_dir = os.path.join(USERS_DIR, str(user_id))
        if not os.path.exists(user_dir):
            return jsonify({"success": False, "message": "User data not found"}), 404
            
        with open(os.path.join(user_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        face_path = os.path.join(user_dir, 'face.jpg')
        with open(face_path, 'rb') as f:
            face_image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        return jsonify({
            "success": True,
            "user_id": user_id,
            "metadata": metadata,
            "face_image": f"data:image/jpeg;base64,{face_image_base64}"
        })
    except Exception as e:
        logger.error(f"Error retrieving user data: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

# Flask API endpoints
@app.route('/signup', methods=['POST'])
def signup():
    global conn
    try:
        data = request.json
        logger.debug(f"Received signup data: user_id={data.get('user_id')}, "
                     f"face_image_len={len(data.get('face_image', ''))}, "
                     f"voice_data_len={len(data.get('voice_data', ''))}")
        user_id = data.get('user_id')
        face_image_base64 = data.get('face_image')
        voice_data_base64 = data.get('voice_data')

        if not user_id or not face_image_base64 or not voice_data_base64:
            logger.error("Missing required fields")
            return jsonify({"success": False, "message": "Missing user_id, face_image, or voice_data"}), 400

        logger.debug("Processing face image")
        face_image_np = process_base64_image(face_image_base64)
        if face_image_np is None:
            logger.error("Invalid face image")
            return jsonify({"success": False, "message": "Invalid face image"}), 400

        logger.debug("Decoding voice data")
        if "," in voice_data_base64:
            voice_data_base64 = voice_data_base64.split(",")[1]
        logger.debug(f"Processing voice data base64 (first 50 chars): {voice_data_base64[:50]}")
        try:
            voice_data = base64.b64decode(voice_data_base64)
        except Exception as e:
            logger.error(f"Error decoding voice data: {str(e)}")
            return jsonify({"success": False, "message": "Invalid voice data format"}), 400

        logger.debug("Saving voice data to file")
        voice_file = process_audio_file(voice_data)
        if voice_file is None:
            logger.error("Invalid voice data")
            return jsonify({"success": False, "message": "Failed to process voice data. Please upload a valid audio file."}), 400

        logger.debug("Calling enroll_user")
        success, message = enroll_user(user_id, face_image_np, voice_file)
        logger.debug(f"Signup result: success={success}, message={message}")
        return jsonify({"success": success, "message": message})
    except Exception as e:
        logger.error(f"Error in signup endpoint: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login():
    global conn
    try:
        data = request.json
        logger.debug(f"Received login data: user_id={data.get('user_id')}, "
                     f"face_image_len={len(data.get('face_image', ''))}, "
                     f"voice_data_len={len(data.get('voice_data', ''))}")
        user_id = data.get('user_id')
        face_image_base64 = data.get('face_image')
        voice_data_base64 = data.get('voice_data')

        if not user_id or not face_image_base64 or not voice_data_base64:
            logger.error("Missing required fields")
            return jsonify({"success": False, "message": "Missing user_id, face_image, or voice_data"}), 400

        logger.debug("Processing face image")
        face_image_np = process_base64_image(face_image_base64)
        if face_image_np is None:
            logger.error("Invalid face image")
            return jsonify({"success": False, "message": "Invalid face image"}), 400

        logger.debug("Decoding voice data")
        if "," in voice_data_base64:
            voice_data_base64 = voice_data_base64.split(",")[1]
        logger.debug(f"Processing voice data base64 (first 50 chars): {voice_data_base64[:50]}")
        try:
            voice_data = base64.b64decode(voice_data_base64)
        except Exception as e:
            logger.error(f"Error decoding voice data: {str(e)}")
            return jsonify({"success": False, "message": "Invalid voice data format"}), 400

        logger.debug("Saving voice data to file")
        voice_file = process_audio_file(voice_data)
        if voice_file is None:
            logger.error("Invalid voice data")
            return jsonify({"success": False, "message": "Failed to process voice data. Please upload a valid audio file."}), 400

        logger.debug("Calling authenticate_user")
        success, message = authenticate_user(user_id, face_image_np, voice_file)
        logger.debug(f"Login result: success={success}, message={message}")
        return jsonify({"success": success, "message": message})
    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        if 'conn' in globals():
            conn.close()
            logger.debug("Database connection closed")