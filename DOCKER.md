# Running IDguard PRO with Docker

This is the fastest way to get IDguard PRO running, especially if you don't want to manage a Python venv, CUDA wheels, and system packages by hand. Two containers, built from the same image: the recording pipeline and the web dashboard.

**Honest caveat up front:** this setup hasn't been verified against real GPU hardware by me (the assistant that wrote it) — I don't have a GPU or a Docker daemon available while writing this. The YAML structure and logic below have been checked carefully, but please treat the very first run as a real test, not a done deal, and report back anything that doesn't match reality.

## Prerequisites

* **Docker** with the modern `docker compose` (V2) plugin — check with `docker compose version`. If you only have the old standalone `docker-compose` (hyphenated, V1), GPU passthrough below may not work reliably; either upgrade, or switch to the `runtime: nvidia` fallback commented in `docker-compose.yml`.
* **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed on the host — this is what actually lets a container see the GPU at all. Verify it works before touching this project at all:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi
  ```
  If that doesn't show your GPU, nothing below will work either — fix that first.
* A locally hosted [Ollama](https://ollama.com) instance if you want AI scene descriptions — this project's Docker setup does **not** include Ollama; point `OLLAMA_URL` in the dashboard Settings at wherever yours runs.

## Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/axeljerabek/IDguard_PRO
   cd IDguard_PRO
   ```

2. **Create your config:**
   ```bash
   cp config.py.example config.py
   ```
   Edit `config.py` — at minimum, set your camera `STREAMS`.

3. **Pre-create the files Docker needs to bind-mount as files, not folders.** This is the single most important step and the easiest one to skip. If a bind-mounted host path doesn't exist yet, Docker creates a **directory** there instead of a file — and the app will fail to open it (or, for the YOLO model file specifically, silently skip its own auto-download, since `config.py`'s download check now treats a real empty file correctly, but a whole *directory* where a file was expected is a different, harder failure).
   ```bash
   touch pipeline_settings.json stream_overrides.json search_index.db
   touch yolo26x.pt   # match this to whatever YOLO_VERSION/MODEL_SIZE you set in config.py
   mkdir -p alerts logs
   ```
   The YOLO model file can stay empty (0 bytes) — the pipeline will detect that and download it properly on first start. `pipeline_settings.json`/`stream_overrides.json` will be filled in the first time you save Settings in the dashboard; `search_index.db` is filled in automatically once search indexing runs.

4. **Build and start:**
   ```bash
   docker compose up -d --build
   ```
   First start downloads the base CUDA image, installs everything, and then downloads the YOLO model from inside the container — this can take a while depending on your connection. Watch progress with:
   ```bash
   docker compose logs -f idguard-pipeline
   ```

5. **Open the dashboard:** `http://<host-ip>:19473`

## Notes specific to this setup

* **`network_mode: host`** is used for both containers — cameras are typically local RTMP streams on your home network, and host networking avoids extra port-mapping/firewall fiddling. This does mean the containers share the host's network namespace directly (fine for a home server, worth knowing if you're hardening a shared machine).
* **NVENC/NVDEC inside the container** (hardware video encode/decode, separate from CUDA compute) depends on whether the `ffmpeg` build inside the image actually has NVENC support compiled in, which varies by Ubuntu version — this is not guaranteed by this Dockerfile. If it doesn't work, the pipeline already has an automatic, self-healing fallback to software encode/decode built in (from long before this Docker setup existed) — you'd just see somewhat higher CPU use per stream than the bare-metal install, not a broken pipeline.
* **Model/embedding caches** (CLAP, sentence-transformers) are kept in named Docker volumes (`model-cache`, `ultralytics-cache`), not bind-mounted — so they survive container restarts and rebuilds without you needing to manage the exact cache folder structure on the host.
* **File ownership:** the containers run as root by default (common for GPU workloads, since `/dev/nvidia*` access typically needs it), so files written into `alerts/` and `logs/` on the host will be root-owned. If that's a problem for you, you'll need to add your own `PUID`/`PGID` handling — not included here, to keep the initial setup simple.
* **Updating:** `git pull`, then `docker compose up -d --build` again. Your `alerts/`, `logs/`, settings files, and model caches are untouched — everything that matters is either bind-mounted or in a named volume, never baked into the image layer.

## Uninstalling / starting fresh

```bash
docker compose down          # stop and remove containers, keep volumes and bind-mounted data
docker compose down -v       # also remove the named volumes (model caches — re-downloads next time)
```
Your `alerts/`, `logs/`, and config files are on the host filesystem regardless — neither command touches those.
