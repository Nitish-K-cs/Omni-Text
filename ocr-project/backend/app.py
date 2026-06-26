
from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import os
import sqlite3
from datetime import datetime
from db_setup import get_db, init_db

app = Flask(__name__)
CORS(app)

# OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize DB
init_db()

# --- SIGNUP ---
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

# --- LOGIN ---
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

# --- OCR + SAVE HISTORY ---
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    username = request.form.get("username", "guest")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = reader.readtext(path)
    texts = [r[1] for r in result]
    confs = [r[2] for r in result]

    final_text = " ".join(texts)
    avg_conf = sum(confs)/len(confs) if confs else 0

    if username != "guest":
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO history (user, image, text, confidence, date)
            VALUES (?, ?, ?, ?, ?)
        """, (username, file.filename, final_text, round(avg_conf, 2),
              datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        conn.close()

    return jsonify({"text": final_text, "confidence": round(avg_conf, 2)})

# --- HISTORY ---
@app.route('/history/<username>')
def history(username):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE user=? ORDER BY id DESC", (username,))
    rows = c.fetchall()
    conn.close()

    return jsonify([dict(row) for row in rows])

# --- CLEAR ALL HISTORY ---
@app.route('/history/clear/<username>', methods=['DELETE'])
def clear_history(username):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user=?", (username,))
    conn.commit()
    conn.close()
    return jsonify({"msg": "History cleared"})

# --- CLEAR SINGLE ENTRY ---
@app.route('/history/delete/<int:entry_id>', methods=['DELETE'])
def delete_history(entry_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()
    return jsonify({"msg": "Entry deleted"})

if __name__ == "__main__":
    app.run(debug=True)              