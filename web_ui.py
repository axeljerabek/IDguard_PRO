from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, Response
import os
import json
import subprocess
import glob
from datetime import datetime

# Lade Konfiguration direkt
from config import (STREAMS, OVERRIDE_F, SETTINGS_F, ALERTS_DIR, PROJECT_ROOT,
                    PRE_ROLL_SEC, POST_ROLL_SEC, TARGET_FPS, CONFIDENCE_THRESHOLD, DETECTION_CLASSES)

app = Flask(__name__)

AVAILABLE_CLASSES = {
    0: "Mensch", 1: "Fahrrad", 2: "Auto", 3: "Motorrad", 4: "Flugzeug", 5: "Bus",
    6: "Zug", 7: "LKW", 8: "Boot", 9: "Ampel", 10: "Feuerhydrant", 11: "Stoppschild",
    12: "Parkuhr", 13: "Bank", 14: "Vogel", 15: "Katze", 16: "Hund", 17: "Pferd",
    18: "Schaf", 19: "Kuh", 20: "Elefant", 21: "Bär", 22: "Zebra", 23: "Giraffe",
    24: "Rucksack", 25: "Regenschirm", 26: "Handtasche", 27: "Krawatte", 28: "Koffer",
    29: "Frisbee", 30: "Skier", 31: "Snowboard", 32: "Sportball", 33: "Drachen",
    34: "Baseballschläger", 35: "Baseballhandschuh", 36: "Skateboard", 37: "Surfbrett",
    38: "Tennisschläger", 39: "Flasche", 40: "Weinglas", 41: "Tasse", 42: "Gabel",
    43: "Messer", 44: "Löffel", 45: "Schüssel", 46: "Banane", 47: "Apfel", 48: "Sandwich",
    49: "Orange", 50: "Brokkoli", 51: "Karotte", 52: "Hotdog", 53: "Pizza", 54: "Donut",
    55: "Kuchen", 56: "Stuhl", 57: "Couch", 58: "Topfpflanze", 59: "Bett", 60: "Esstisch",
    61: "Toilette", 62: "Fernseher", 63: "Laptop", 64: "Maus", 65: "Fernbedienung",
    66: "Tastatur", 67: "Handy", 68: "Mikrowelle", 69: "Backofen", 70: "Toaster",
    71: "Spülbecken", 72: "Kühlschrank", 73: "Buch", 74: "Uhr", 75: "Vase", 76: "Schere",
    77: "Teddybär", 78: "Föhn", 79: "Zahnbürste"
}

def is_pipeline_running():
    result = subprocess.run(['pgrep', '-f', 'recorder_pipeline.py'], capture_output=True)
    return result.returncode == 0

def load_overrides():
    overrides = {s["name"]: ("ON" if s["enabled"] else "OFF") for s in STREAMS}
    if os.path.exists(OVERRIDE_F):
        try:
            with open(OVERRIDE_F, 'r') as f:
                overrides.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return overrides

def load_settings():
    settings = {
        "PRE_ROLL_SEC": PRE_ROLL_SEC, "POST_ROLL_SEC": POST_ROLL_SEC,
        "TARGET_FPS": TARGET_FPS, "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "DETECTION_CLASSES": DETECTION_CLASSES
    }
    if os.path.exists(SETTINGS_F):
        try:
            with open(SETTINGS_F, 'r') as f:
                settings.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return settings

def save_overrides(data):
    with open(OVERRIDE_F, 'w') as f:
        json.dump(data, f, indent=4)

def format_size(size_bytes):
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"

@app.route('/')
def dashboard():
    overrides = load_overrides()
    settings = load_settings()
    pipeline_active = is_pipeline_running()

    alerts = sorted(glob.glob(os.path.join(ALERTS_DIR, '*.mp4')), key=os.path.getmtime, reverse=True)
    recent_events = []
    for alert in alerts:
        try:
            mtime = os.path.getmtime(alert)
            size = os.path.getsize(alert)
            dt_str = datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M')
            recent_events.append({
                'filename': os.path.basename(alert),
                'datetime': dt_str,
                'size': format_size(size)
            })
        except OSError:
            pass

    streams = [s["name"] for s in STREAMS]

    template = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IDguard PRO</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg: #0d1117; --surface: rgba(33, 38, 45, 0.7); --surface-hover: rgba(48, 54, 61, 0.9);
                --primary: #2f81f7; --primary-hover: #388bfd;
                --success: #2ea043; --danger: #f85149; --text: #c9d1d9;
                --text-muted: #8b949e; --border: #30363d; --input-bg: #010409;
            }
            body {
                font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text);
                margin: 0; padding: 2rem; background-image: radial-gradient(circle at top right, #1f2937, #0d1117);
                min-height: 100vh;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { font-weight: 300; margin-bottom: 2rem; display: flex; align-items: center; gap: 10px; }
            h1 i { color: var(--primary); }
            h2 { font-weight: 500; font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;}

            .status-badge { font-size: 0.75rem; padding: 3px 10px; border-radius: 12px; margin-left: auto; text-transform: uppercase; font-weight: 600; }
            .status-active { background: var(--success); color: white; }
            .status-inactive { background: var(--border); color: var(--text-muted); }

            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
            @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

            .card {
                background: var(--surface); backdrop-filter: blur(10px); border-radius: 12px;
                padding: 1.5rem; border: 1px solid var(--border); margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }

            button {
                padding: 10px 16px; cursor: pointer; border: none; border-radius: 6px;
                font-weight: 500; transition: all 0.2s ease; font-family: 'Inter', sans-serif;
                display: flex; align-items: center; justify-content: center; gap: 8px;
            }
            button:hover { filter: brightness(1.1); transform: translateY(-1px); }
            .btn-start { background: var(--success); color: white; flex: 1; }
            .btn-stop { background: var(--danger); color: white; flex: 1; }
            .btn-save { background: var(--primary); color: white; width: 100%; margin-top: 20px; font-size: 1rem; padding: 12px; }
            .btn-delete { background: transparent; color: var(--danger); border: 1px solid var(--danger); padding: 6px 12px; font-size: 0.8rem; border-radius: 4px; }
            .btn-delete:hover { background: var(--danger); color: white; }

            .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
            .switch input { opacity: 0; width: 0; height: 0; }
            .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: var(--border); transition: .3s; border-radius: 24px; }
            .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; }
            input:checked + .slider { background-color: var(--success); }
            input:checked + .slider:before { transform: translateX(20px); }

            .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            .settings-group { display: flex; flex-direction: column; }
            .settings-group label { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
            .settings-group input { padding: 10px; background: var(--input-bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; font-size: 0.95rem; font-family: 'Inter', sans-serif; }
            .settings-group input:focus { border-color: var(--primary); outline: none; box-shadow: 0 0 0 2px rgba(47, 129, 247, 0.2); }

            .range-wrap { display: flex; align-items: center; gap: 10px; }
            .range-wrap input[type=range] { flex: 1; -webkit-appearance: none; background: transparent; padding: 0; border: none; }
            .range-wrap input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%; background: var(--primary); cursor: pointer; margin-top: -6px; }
            .range-wrap input[type=range]::-webkit-slider-runnable-track { width: 100%; height: 4px; cursor: pointer; background: var(--border); border-radius: 2px; }
            .range-val { background: var(--input-bg); padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; border: 1px solid var(--border); width: 35px; text-align: center; }

            .multiselect { position: relative; width: 100%; }
            .selectBox select { width: 100%; padding: 10px; background: var(--input-bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; font-size: 0.95rem; cursor: pointer; appearance: none; }
            .selectBox::after { content: "\\f0d7"; font-family: "Font Awesome 5 Free"; font-weight: 900; position: absolute; right: 15px; top: 10px; color: var(--text-muted); pointer-events: none; }
            .overSelect { position: absolute; left: 0; right: 0; top: 0; bottom: 0; cursor: pointer; }
            #checkboxes { display: none; position: absolute; background: var(--surface); backdrop-filter: blur(15px); border: 1px solid var(--border); border-radius: 6px; width: 100%; max-height: 300px; overflow-y: auto; z-index: 100; box-shadow: 0 10px 30px rgba(0,0,0,0.8); margin-top: 5px; }
            #checkboxes label { display: flex; align-items: center; gap: 10px; padding: 10px 15px; cursor: pointer; color: var(--text); border-bottom: 1px solid var(--border); margin: 0; text-transform: none; font-size: 0.9rem; transition: background 0.2s; }
            #checkboxes label:hover { background-color: var(--primary); color: white; }

            .event-list { max-height: 600px; overflow-y: auto; padding-right: 5px; }
            .event-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(255,255,255,0.02); margin-bottom: 8px; border-radius: 8px; border: 1px solid var(--border); transition: all 0.2s; cursor: pointer; }
            .event-item:hover { background: var(--surface-hover); border-color: var(--primary); transform: translateX(2px); }
            .event-info { display: flex; flex-direction: column; gap: 4px; }
            .event-name { font-weight: 500; font-size: 0.95rem; color: var(--primary); }
            .event-meta { color: var(--text-muted); font-size: 0.8rem; display: flex; gap: 10px; }

            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

            .lightbox { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); backdrop-filter: blur(5px); justify-content: center; align-items: center; flex-direction: column; opacity: 0; transition: opacity 0.3s ease; }
            .lightbox.active { display: flex; opacity: 1; }
            .lightbox-content { position: relative; max-width: 90%; max-height: 85%; width: 1000px; transform: scale(0.95); transition: transform 0.3s ease; }
            .lightbox.active .lightbox-content { transform: scale(1); }
            video { width: 100%; border-radius: 10px; background: #000; box-shadow: 0 10px 40px rgba(0,0,0,0.5); border: 1px solid var(--border); }
            .close-btn { position: absolute; top: -40px; right: 0; color: white; font-size: 28px; cursor: pointer; transition: color 0.2s; }
            .close-btn:hover { color: var(--danger); }
            .player-controls { display: flex; gap: 8px; justify-content: center; margin-top: 15px; }
            .player-controls button { background: var(--input-bg); color: var(--text); border: 1px solid var(--border); border-radius: 20px; padding: 6px 16px; font-size: 0.85rem; }
            .player-controls button.active { background: var(--primary); color: white; border-color: var(--primary); }

            .loader { border: 3px solid var(--border); border-top: 3px solid var(--primary); border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; display: none; margin: 10px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            #status-text { font-size: 0.85rem; color: var(--text-muted); text-align: center; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><i class="fa-solid fa-shield-halved"></i> IDguard PRO Dashboard</h1>

            <div class="grid">
                <!-- Linke Spalte -->
                <div>
                    <div class="card">
                        <h2>
                            <i class="fa-solid fa-power-off"></i> Pipeline Control
                            <span id="pipeline-status" class="status-badge {{ 'status-active' if pipeline_active else 'status-inactive' }}">
                                {{ 'Aktiv' if pipeline_active else 'Inaktiv' }}
                            </span>
                        </h2>
                        <div style="display: flex; gap: 15px;">
                            <form action="/start" method="POST" style="flex:1; margin:0;">
                                <button type="submit" class="btn-start"><i class="fa-solid fa-play"></i> START</button>
                            </form>
                            <form action="/stop" method="POST" style="flex:1; margin:0;">
                                <button type="submit" class="btn-stop"><i class="fa-solid fa-stop"></i> STOP</button>
                            </form>
                        </div>
                    </div>

                    <div class="card">
                        <h2><i class="fa-solid fa-video"></i> Kamera Aktivierung</h2>
                        <div>
                            {% for stream in streams %}
                            <div class="stream-item" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="font-weight: 500;">{{ stream }}</span>
                                <form action="/toggle/{{ stream }}" method="POST" style="margin: 0;">
                                    <label class="switch">
                                        <input type="checkbox" onchange="this.form.submit()" {% if overrides.get(stream, 'ON') == 'ON' %}checked{% endif %}>
                                        <span class="slider"></span>
                                    </label>
                                </form>
                            </div>
                            {% endfor %}
                        </div>
                    </div>

                    <div class="card">
                        <h2><i class="fa-solid fa-sliders"></i> System Einstellungen</h2>
                        <form action="/save_settings" method="POST">
                            <div class="settings-grid">
                                <div class="settings-group">
                                    <label>Target FPS</label>
                                    <input type="number" name="TARGET_FPS" value="{{ settings.TARGET_FPS }}" required>
                                </div>
                                <div class="settings-group">
                                    <label>Confidence Threshold</label>
                                    <div class="range-wrap">
                                        <input type="range" min="0.1" max="1.0" step="0.05" name="CONFIDENCE_THRESHOLD" value="{{ settings.CONFIDENCE_THRESHOLD }}" oninput="document.getElementById('conf_val').innerText = this.value">
                                        <span id="conf_val" class="range-val">{{ settings.CONFIDENCE_THRESHOLD }}</span>
                                    </div>
                                </div>
                                <div class="settings-group">
                                    <label>Pre-Roll (Sek)</label>
                                    <input type="number" name="PRE_ROLL_SEC" value="{{ settings.PRE_ROLL_SEC }}" required>
                                </div>
                                <div class="settings-group">
                                    <label>Post-Roll (Sek)</label>
                                    <input type="number" name="POST_ROLL_SEC" value="{{ settings.POST_ROLL_SEC }}" required>
                                </div>
                            </div>

                            <div class="settings-group" style="margin-top: 15px;">
                                <label>Objekt-Kategorien (YOLOv10)</label>
                                <div class="multiselect">
                                    <div class="selectBox" onclick="toggleCheckboxes()">
                                        <select>
                                            <option id="category-summary">Lade Kategorien...</option>
                                        </select>
                                        <div class="overSelect"></div>
                                    </div>
                                    <div id="checkboxes">
                                        {% for class_id, class_name in available_classes.items() %}
                                            <label for="cls_{{ class_id }}">
                                                <input type="checkbox" id="cls_{{ class_id }}" name="DETECTION_CLASSES" value="{{ class_id }}"
                                                {% if class_id in settings.DETECTION_CLASSES %}checked{% endif %}
                                                onchange="updateCategorySummary()" />
                                                {{ class_name }}
                                            </label>
                                        {% endfor %}
                                    </div>
                                </div>
                            </div>

                            <button type="submit" class="btn-save"><i class="fa-solid fa-floppy-disk"></i> Einstellungen anwenden</button>
                        </form>
                    </div>
                </div>

                <!-- Rechte Spalte -->
                <div>
                    <div class="card" style="height: calc(100% - 3rem);">
                        <h2>
                            <i class="fa-solid fa-clapperboard"></i> Letzte Aufzeichnungen
                            <span id="event-count" style="background: var(--border); padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-left: 10px;">{{ recent_events|length }}</span>
                        </h2>
                        <div class="event-list">
                            {% for event in recent_events %}
                            <div class="event-item" onclick="openLightbox('{{ event.filename }}')">
                                <div class="event-info">
                                    <span class="event-name"><i class="fa-regular fa-file-video" style="margin-right: 5px;"></i> {{ event.filename }}</span>
                                    <span class="event-meta">
                                        <span><i class="fa-regular fa-calendar"></i> {{ event.datetime }}</span>
                                        <span><i class="fa-solid fa-hard-drive"></i> {{ event.size }}</span>
                                    </span>
                                </div>
                                <form action="/delete/{{ event.filename }}" method="POST" style="margin: 0;" onsubmit="return confirm('Video wirklich löschen?');">
                                    <button type="submit" class="btn-delete" title="Video löschen" onclick="event.stopPropagation()">
                                        <i class="fa-solid fa-trash-can"></i>
                                    </button>
                                </form>
                            </div>
                            {% endfor %}
                            {% if not recent_events %}
                            <div style="text-align: center; padding: 3rem 0; color: var(--text-muted);">
                                <i class="fa-solid fa-film" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                                <p>Bisher keine Events aufgezeichnet.</p>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
            <div class="lightbox-content" onclick="event.stopPropagation()">
                <span class="close-btn" onclick="closeLightbox(event)"><i class="fa-solid fa-xmark"></i></span>
                <video id="videoPlayer" controls muted><source id="videoSource" src="" type="video/mp4"></video>
                <div id="loader" class="loader"></div>
                <div id="status-text"></div>
                <div class="player-controls">
                    <button onclick="setSpeed(0.5, this)">0.5x</button>
                    <button onclick="setSpeed(1.0, this)" class="active" id="defaultSpeedBtn">1.0x</button>
                    <button onclick="setSpeed(2.0, this)">2.0x</button>
                    <button onclick="setSpeed(4.0, this)">4.0x</button>
                </div>
            </div>
        </div>

        <script>
            // --- UI-Logik ---
            let expanded = false;
            function toggleCheckboxes() {
                const checkboxes = document.getElementById("checkboxes");
                expanded = !expanded;
                checkboxes.style.display = expanded ? "block" : "none";
            }

            function updateCategorySummary() {
                const checkedBoxes = document.querySelectorAll('#checkboxes input[type="checkbox"]:checked');
                const summary = document.getElementById("category-summary");
                if (checkedBoxes.length === 0) { summary.innerText = "Keine Klassen gewählt"; }
                else if (checkedBoxes.length === 1) { summary.innerText = "1 Klasse ausgewählt"; }
                else { summary.innerText = checkedBoxes.length + " Klassen ausgewählt"; }
            }
            updateCategorySummary();

            document.addEventListener('click', function(event) {
                const multiselect = document.querySelector('.multiselect');
                if (expanded && !multiselect.contains(event.target)) {
                    document.getElementById("checkboxes").style.display = "none";
                    expanded = false;
                }
            });

            // --- Auto-Refresh (Hintergrund-Update) ---
            function autoRefresh() {
                fetch(window.location.href)
                    .then(response => response.text())
                    .then(html => {
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');

                        // Status aktualisieren
                        const newStatus = doc.querySelector('#pipeline-status');
                        const statusBadge = document.querySelector('#pipeline-status');
                        if (newStatus && statusBadge) {
                            statusBadge.className = newStatus.className;
                            statusBadge.innerHTML = newStatus.innerHTML;
                        }

                        // Event-Liste aktualisieren
                        const newEvents = doc.querySelector('.event-list');
                        if (newEvents) {
                            document.querySelector('.event-list').innerHTML = newEvents.innerHTML;
                        }

                        // Event-Counter aktualisieren
                        const newCount = doc.querySelector('#event-count');
                        const countBadge = document.querySelector('#event-count');
                        if (newCount && countBadge) {
                            countBadge.innerHTML = newCount.innerHTML;
                        }
                    })
                    .catch(err => console.error('Refresh fehlgeschlagen:', err));
            }
            // Alle 10 Sekunden automatisch aktualisieren
            setInterval(autoRefresh, 10000);

            // --- Lightbox-Logik ---
            const lightbox = document.getElementById('lightbox'),
                  videoPlayer = document.getElementById('videoPlayer'),
                  videoSource = document.getElementById('videoSource'),
                  speedBtns = document.querySelectorAll('.player-controls button'),
                  statusText = document.getElementById('status-text'),
                  loader = document.getElementById('loader');

            function openLightbox(filename) {
                loader.style.display = 'block';
                statusText.innerText = 'Lade & transkodiere Video...';
                videoSource.src = '/video/' + filename;
                videoPlayer.load();

                videoPlayer.onloadeddata = () => {
                    loader.style.display = 'none';
                    statusText.innerText = '';
                    videoPlayer.play().catch(e => { statusText.innerText = 'Wiedergabe pausiert. Klicke Play.'; });
                };
                videoPlayer.onerror = () => {
                    loader.style.display = 'none';
                    statusText.innerText = 'Fehler beim Laden der Datei.';
                };

                setSpeed(1.0, document.getElementById('defaultSpeedBtn'));
                lightbox.classList.add('active');
            }

            function closeLightbox(e) {
                videoPlayer.pause();
                videoSource.src = '';
                videoPlayer.load();
                lightbox.classList.remove('active');
            }

            function setSpeed(rate, btn) {
                videoPlayer.playbackRate = rate;
                speedBtns.forEach(b => b.classList.remove('active'));
                if (btn) btn.classList.add('active');
            }

            document.addEventListener('keydown', function(e) {
                if (e.key === "Escape" && lightbox.classList.contains('active')) closeLightbox();
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(template, streams=streams, overrides=overrides, settings=settings, available_classes=AVAILABLE_CLASSES, recent_events=recent_events, pipeline_active=pipeline_active)

@app.route('/start', methods=['POST'])
def start_pipeline():
    subprocess.Popen(['/bin/bash', os.path.join(PROJECT_ROOT, 'start_detached.sh')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return redirect(url_for('dashboard'))

@app.route('/stop', methods=['POST'])
def stop_pipeline():
    subprocess.run(['/bin/bash', os.path.join(PROJECT_ROOT, 'stop.sh')])
    return redirect(url_for('dashboard'))

@app.route('/toggle/<name>', methods=['POST'])
def toggle_stream(name):
    overrides = load_overrides()
    overrides[name] = 'ON' if overrides.get(name, 'OFF') == 'OFF' else 'OFF'
    save_overrides(overrides)
    return redirect(url_for('dashboard'))

@app.route('/save_settings', methods=['POST'])
def save_pipeline_settings():
    settings = {
        "TARGET_FPS": int(request.form.get('TARGET_FPS', 30)),
        "CONFIDENCE_THRESHOLD": float(request.form.get('CONFIDENCE_THRESHOLD', 0.5)),
        "PRE_ROLL_SEC": int(request.form.get('PRE_ROLL_SEC', 10)),
        "POST_ROLL_SEC": int(request.form.get('POST_ROLL_SEC', 30)),
        "DETECTION_CLASSES": [int(x) for x in request.form.getlist('DETECTION_CLASSES')] or [0]
    }
    with open(SETTINGS_F, 'w') as f:
        json.dump(settings, f, indent=4)
    return redirect(url_for('dashboard'))

@app.route('/delete/<filename>', methods=['POST'])
def delete_video(filename):
    file_path = os.path.abspath(os.path.join(ALERTS_DIR, filename))
    if file_path.startswith(os.path.abspath(ALERTS_DIR)) and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Fehler beim Löschen: {e}")
    return redirect(url_for('dashboard'))

@app.route('/video/<filename>')
def serve_annot_video(filename):
    input_file = os.path.join(ALERTS_DIR, filename)
    if not os.path.exists(input_file): return f"File not found: {filename}", 404

    def generate():
        process = subprocess.Popen(['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '28', '-c:a', 'mp3', '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1)
        try:
            while True:
                chunk = process.stdout.read(8192)
                if not chunk: break
                yield chunk
        except Exception as e:
            process.kill()
        finally:
            process.stdout.close()
    return Response(generate(), mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=19473)
