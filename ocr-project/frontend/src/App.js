import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [page, setPage] = useState("home");
  const [user, setUser] = useState(null);
  const [view, setView] = useState("dashboard");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [confidence, setConfidence] = useState("");
  const [history, setHistory] = useState([]);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [form, setForm] = useState({ username: "", password: "" });

  const resetState = () => { setFile(null); setPreview(null); setText(""); setConfidence(""); setHistory([]); };

  const login = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (res.ok) {
        resetState(); setUser(form.username); setPage("dashboard");
        setTimeout(() => fetchHistory(form.username), 300);
      } else { alert(data.msg); }
    } catch { alert("Backend not reachable"); }
  };

  const signup = async () => {
    await fetch("http://127.0.0.1:5000/signup", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    alert("Account created — you can now sign in.");
  };

  const fetchHistory = async (username) => {
    try {
      const res = await fetch(`http://127.0.0.1:5000/history/${username}`);
      setHistory(await res.json());
    } catch (err) { console.log(err); }
  };

  const clearAllHistory = async () => {
    await fetch(`http://127.0.0.1:5000/history/clear/${user}`, { method: "DELETE" });
    fetchHistory(user);
  };

  const clearEntry = async (id) => {
    await fetch(`http://127.0.0.1:5000/history/delete/${id}`, { method: "DELETE" });
    fetchHistory(user);
  };

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream; streamRef.current = stream;
  };
  const stopCamera = () => streamRef.current?.getTracks().forEach((t) => t.stop());
  const capture = () => {
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      const f = new File([blob], "capture.png");
      setFile(f); setPreview(URL.createObjectURL(blob));
    });
    stopCamera();
  };
  const handleFile = (e) => { const f = e.target.files[0]; setFile(f); setPreview(URL.createObjectURL(f)); };
  const extract = async () => {
    if (!file) return alert("Select an image first");
    const fd = new FormData();
    fd.append("image", file); fd.append("username", user || "guest");
    const res = await fetch("http://127.0.0.1:5000/upload", { method: "POST", body: fd });
    const data = await res.json();
    setText(data.text); setConfidence(data.confidence);
    if (user) fetchHistory(user);
  };

  // ── RESULT BLOCK (reused across views) ──────────────────
  const ResultBlock = () => (text || confidence) ? (
    <div className="result-box">
      <div className="result-box-header">
        <div className="result-box-header-dot" />
        <span className="result-label">Extracted text</span>
      </div>
      <div className="result-box-body">
        <p>{text || "—"}</p>
        {confidence && <div className="confidence-badge">{confidence} confidence</div>}
      </div>
    </div>
  ) : null;

  // ══════════════════════════════════════════════════════
  //  HOME PAGE
  // ══════════════════════════════════════════════════════
  if (page === "home") {
    return (
      <div className="center">
        {/* Left hero */}
        <div className="auth-hero">
          <div className="auth-hero-badge">OMNITEXT-SMART TEXT DETECTION SYSTEM</div>
          <h1>
            Extract text from<br />
            <span>any image instantly</span>
          </h1>
          <p>
            OmniText uses advanced optical character recognition to pull
            text from photos, screenshots, scanned documents, and more.
          </p>
          <div className="hero-features">
            {["Upload or capture with your camera", "High-accuracy text extraction", "Full extraction history"].map((f) => (
              <div className="hero-feature" key={f}>
                <div className="hero-feature-dot" />
                {f}
              </div>
            ))}
          </div>
        </div>

        {/* Right form */}
        <div className="auth-form-panel">
          <div className="auth-form-inner">
            <h2>Welcome</h2>
            <p>Sign in to your OmniText account</p>
            <div className="auth-card">
              <div className="input-group">
                <label>Username</label>
                <input placeholder="you@example.com" value={form.username}
                  onChange={(e) => setForm({ ...form, username: e.target.value })}
                  onKeyDown={(e) => e.key === "Enter" && login()} />
              </div>
              <div className="input-group">
                <label>Password</label>
                <input type="password" placeholder="••••••••" value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  onKeyDown={(e) => e.key === "Enter" && login()} />
              </div>
              <button className="btn-primary" onClick={login}>Sign in</button>
              <div className="auth-divider">or</div>
              <button className="btn-ghost" onClick={signup}>Create account</button>
              <button className="btn-ghost" onClick={() => { resetState(); setUser(null); setPage("guest"); }}>
                Continue as guest
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ══════════════════════════════════════════════════════
  //  GUEST PAGE
  // ══════════════════════════════════════════════════════
  if (page === "guest") {
    return (
      <div className="guest-page">
        <h2>Try OmniText</h2>
        <div className="controls">
          <label className="upload-label">
            ↑ Choose file
            <input type="file" onChange={handleFile} />
          </label>
          <button className="primary" onClick={extract}>Extract text</button>
        </div>
        <div className="scan-frame" style={{ width: 300, height: 210 }}>
          <div className="scanline" />
          <div className="scan-corner tl" /><div className="scan-corner tr" />
          <div className="scan-corner bl" /><div className="scan-corner br" />
          {preview ? <img src={preview} alt="preview" /> : <span>No image selected</span>}
        </div>
        <ResultBlock />
        <button className="btn-ghost" style={{ marginTop: 4 }}
          onClick={() => { resetState(); setPage("home"); }}>
          ← Back to sign in
        </button>
      </div>
    );
  }

  // ══════════════════════════════════════════════════════
  //  DASHBOARD
  // ══════════════════════════════════════════════════════
  const navItems = [
    { id: "dashboard", label: "Dashboard",  icon: "⊞" },
    { id: "camera",    label: "Camera OCR", icon: "◉" },
    { id: "history",   label: "History",    icon: "◷" },
  ];

  return (
    <div className="app">
      {/* SIDEBAR */}
      <div className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">OT</div>
          <span className="sidebar-brand-name">OmniText</span>
        </div>

        <div className="sidebar-user">
          <h2>Signed in as</h2>
          <div className="sidebar-user-chip">
            <div className="sidebar-user-avatar">{user?.[0]}</div>
            <span className="sidebar-user-name">{user}</span>
          </div>
        </div>

        <ul>
          {navItems.map((item) => (
            <li key={item.id} className={view === item.id ? "active" : ""}
              onClick={() => {
                resetState(); setView(item.id);
                if (item.id === "history") setTimeout(() => fetchHistory(user), 200);
              }}>
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </li>
          ))}
        </ul>

        <div className="sidebar-bottom">
          <ul>
            <li onClick={() => { resetState(); setUser(null); setPage("home"); }}>
              <span className="nav-icon">→</span> Sign out
            </li>
          </ul>
        </div>
      </div>

      {/* MAIN */}
      <div className="main">

        {/* ── DASHBOARD ── */}
        {view === "dashboard" && (
          <div className="card">
            <div className="page-header">
              <h2>Upload image</h2>
              <p>Upload a photo or document to extract its text</p>
            </div>
            <div className="controls">
              <label className="upload-label">
                ↑ Choose file
                <input type="file" onChange={handleFile} />
              </label>
              <button className="primary" onClick={extract}>Extract text</button>
            </div>
            <div className="preview-box">
              <div className="scan-frame">
                <div className="scanline" />
                <div className="scan-corner tl" /><div className="scan-corner tr" />
                <div className="scan-corner bl" /><div className="scan-corner br" />
                {preview ? <img src={preview} alt="preview" /> : <span>No image selected</span>}
              </div>
            </div>
            <ResultBlock />
          </div>
        )}

        {/* ── CAMERA ── */}
        {view === "camera" && (
          <div className="card">
            <div className="page-header">
              <h2>Camera OCR</h2>
              <p>Capture with your camera, then extract text</p>
            </div>
            <div className="controls">
              <button onClick={startCamera}>▶ Start</button>
              <button onClick={stopCamera}>◼ Stop</button>
              <button onClick={capture}>⊙ Capture</button>
              <button className="primary" onClick={extract}>Extract text</button>
            </div>
            <div className="preview-box">
              <video ref={videoRef} autoPlay
                style={{ width: 300, borderRadius: 12, border: "1px solid rgba(255,255,255,.1)" }} />
              {preview && (
                <div className="scan-frame">
                  <div className="scanline" />
                  <div className="scan-corner tl" /><div className="scan-corner tr" />
                  <div className="scan-corner bl" /><div className="scan-corner br" />
                  <img src={preview} alt="captured" />
                </div>
              )}
            </div>
            <ResultBlock />
          </div>
        )}

        {/* ── HISTORY ── */}
        {view === "history" && (
          <div className="card">
            <div className="history-header">
              <div className="page-header" style={{ marginBottom: 0 }}>
                <h2>History</h2>
                <p>{history.length} extraction{history.length !== 1 ? "s" : ""}</p>
              </div>
              {history.length > 0 && (
                <button className="btn-danger-ghost" onClick={clearAllHistory}>Clear all</button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">◷</div>
                <p className="es-title">No extractions yet</p>
                <p className="es-sub">Upload an image to get started</p>
              </div>
            ) : (
              <div className="history-list">
                {history.map((h, i) => (
                  <div key={i} className="history-item">
                    <img src={`http://127.0.0.1:5000/static/uploads/${h.image}`} alt="thumb" />
                    <div className="history-text">
                      <p className="ht-main">{h.text}</p>
                      <p className="ht-date">{h.date}</p>
                    </div>
                    <button className="btn-danger-ghost" onClick={() => clearEntry(h.id)}>Delete</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;