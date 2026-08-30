rm -rf .venv venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fix für einen echten Packaging-Konflikt: insightface/faster-whisper
# hängen selbst am reinen (CPU-only) onnxruntime, das denselben Python-
# Modul-Namespace wie onnxruntime-gpu teilt. Je nachdem, in welcher
# Reihenfolge pip die Pakete oben tatsächlich installiert (nicht
# garantiert dieselbe wie in requirements.txt), kann das CPU-Paket
# zuletzt geschrieben werden und onnxruntime-gpu lautlos überschreiben —
# ohne Fehlermeldung, nur stille CPU-Inferenz für Gesichtserkennung und
# Transkription. Hier direkt danach erzwungen richtiggestellt.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --force-reinstall --no-deps onnxruntime-gpu

echo ""
echo "Verifiziere GPU-Unterstützung für onnxruntime (Gesichtserkennung/Transkription):"
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
