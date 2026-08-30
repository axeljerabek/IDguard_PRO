# IDguard PRO - Container-Image
#
# Basis: nvidia/cuda "runtime" (nicht "devel") — wir kompilieren nichts,
# PyTorch bringt seine eigenen CUDA-Bibliotheken über den cu128-Wheel mit.
# Die tatsächliche GPU-Nutzung passiert über das NVIDIA Container Toolkit
# auf dem Host, nicht durch irgendwas, das hier im Image gebaut wird.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# ffmpeg: wird NUR von web_ui.py für die Video-Wiedergabe-Transcodierung im
# Dashboard gebraucht (subprocess-Aufruf) — die Aufnahme-Pipeline selbst
# nutzt PyAV direkt, braucht das ffmpeg-Binary nicht.
#
# Hinweis: ob das ffmpeg aus den Ubuntu-Paketquellen NVENC/NVDEC-Unterstützung
# mitbringt, hängt von der Ubuntu-Version ab und ist hier nicht garantiert.
# Kein Problem — die Pipeline fällt dann automatisch auf Software-Encoding
# zurück (das war schon vorher eingebaut), nur mit etwas mehr CPU-Last.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Erst nur requirements.txt kopieren, damit Docker den pip-Install-Layer
# cached und nicht bei jeder Code-Änderung neu ausführt.
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Läuft standardmäßig als root im Container (üblich für GPU-Workloads, da
# /dev/nvidia* i.d.R. root-Zugriff braucht) — Volumes (alerts/, logs/) landen
# dadurch auch als root-owned auf dem Host. Für saubere Dateirechte auf dem
# Host ggf. per PUID/PGID + entsprechendem Entrypoint erweitern — hier
# bewusst weggelassen, um die Ersteinrichtung nicht zu verkomplizieren.

EXPOSE 19473
