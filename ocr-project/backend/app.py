from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
from db_setup import get_db, init_db

app = Flask(__name__)
CORS(app)
reader = easyocr.Reader(['en'], gpu=False)

# ================= MODEL =================
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    result = decoded[0].numpy()

    output_text = ""
    for p in result[0]:
        if int(p) != -1:
            output_text += char_list[int(p)]
    return output_text

# Load model safely
try:
    model = tf.keras.models.load_model("Models/nitish.keras", compile=False)
    print("Custom model loaded")
except Exception as e:
    print(" Custom model loaded", e)
    model = None

# ================= UPLOAD =================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Init DB
init_db()

# ================= SIGNUP =================
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    conn = get_db()
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (data['username'], data['password']))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"msg": "User exists"}), 400
    finally:
        conn.close()

    return jsonify({"msg": "Signup success"})

# ================= LOGIN =================
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (data['username'], data['password']))
    user = c.fetchone()
    conn.close()

    if user:
        return jsonify({"msg": "Login success"})
    return jsonify({"msg": "Invalid credentials"}), 401


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    username = request.form.get("username", "guest")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    final_text = ""
    confidence = 0

    
    if model:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (128, 32))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)
                pred_text = decode_prediction(pred)

                final_text = pred_text
                confidence = float(np.max(pred))

        except Exception as e:
            print("Model error:", e)

    
    if confidence < 0.8 or final_text.strip() == "":
        result = reader.readtext(path)
        texts = [r[1] for r in result]
        confs = [r[2] for r in result]

        final_text = " ".join(texts)
        confidence = sum(confs)/len(confs) if confs else 0

    # ===== SAVE HISTORY =====
    if username != "guest":
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO history (user, image, text, confidence, date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            username,
            file.filename,
            final_text,
            round(confidence, 2),
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        conn.commit()
        conn.close()

    return jsonify({
        "text": final_text,
        "confidence": round(confidence, 2)
    })

# ================= HISTORY =================
@app.route('/history/<username>')
def history(username):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE user=? ORDER BY id DESC", (username,))
    rows = c.fetchall()
    conn.close()

    return jsonify([dict(row) for row in rows])

# ================= CLEAR ALL =================
@app.route('/history/clear/<username>', methods=['DELETE'])
def clear_history(username):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user=?", (username,))
    conn.commit()
    conn.close()
    return jsonify({"msg": "History cleared"})

# ================= DELETE ONE =================
@app.route('/history/delete/<int:entry_id>', methods=['DELETE'])
def delete_history(entry_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()
    return jsonify({"msg": "Entry deleted"})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)