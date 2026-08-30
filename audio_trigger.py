"""
audio_trigger.py - Optionaler Audio-Trigger per CLAP (Contrastive Language-Audio
Pretraining, laion/clap-htsat-unfused).

WARUM EIGENER THREAD: CLAP-Inferenz dauert (auf CPU) mehrere hundert ms bis
über eine Sekunde. Würde das im Video-Frame-Loop selbst laufen, würde es die
Aufnahme spürbar verzögern/blockieren — inakzeptabel, da Aufzeichnung Priorität
Nummer 1 hat. Stattdessen: der Haupt-Loop ruft nur `feed()` auf (extrem
billig, nur ein Buffer-Append), ein separater Hintergrund-Thread holt sich
periodisch eine Kopie des Puffers und macht die eigentliche Klassifikation.

MEHRERE FREI WÄHLBARE KATEGORIEN GLEICHZEITIG: anders als YAMNet/PANNs (feste
Klassenliste) vergleicht CLAP eine Audio-Aufnahme mit BELIEBIGEM Text im
selben Embedding-Raum — "whispering", "glass breaking", "dog barking" oder
was auch immer, ohne Neu-Training. Alle konfigurierten Kategorien werden pro
Durchlauf parallel verglichen, die mit der höchsten Ähnlichkeit gewinnt,
sofern sie über der Schwelle liegt.

Fehlt die Abhängigkeit (torch/transformers) oder schlägt der Download fehl,
bleibt das Feature einfach inaktiv (klar geloggt) — nie ein Grund, die
Pipeline zu stören.
"""
import threading
import time

import numpy as np


def _extract_features(output):
    """get_text_features()/get_audio_features() liefern je nach installierter
    transformers-Version entweder direkt ein Tensor ODER ein ModelOutput-
    Objekt (z.B. BaseModelOutputWithPooling) mit dem eigentlichen Tensor als
    Attribut. Robust gegen beide Fälle, statt blind .norm() draufzurufen."""
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "audio_embeds"):
        return output.audio_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "norm"):  # bereits ein rohes Tensor
        return output
    raise TypeError(f"Unerwarteter Rückgabetyp von get_*_features(): {type(output)}")


class AudioTrigger:
    def __init__(self, logger, name, sample_rate=48000, window_sec=3.0):
        self.logger = logger
        self.name = name
        self.sample_rate = sample_rate
        self.window_sec = window_sec

        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.float32)

        self._triggered = False
        self._triggered_label = None

        self._stop = threading.Event()
        self._thread = None

        self._model = None
        self._processor = None
        self._load_failed = False

        self._text_embeds = None
        self._text_labels = []

    # --- Öffentliche Schnittstelle für den Haupt-Loop ---------------------

    def start(self, get_settings):
        """get_settings: Callable ohne Argumente, liefert das aktuelle
        Settings-Dict — wird periodisch neu abgefragt, damit Kategorien,
        Schwelle und Enable/Disable live (ohne Pipeline-Neustart) greifen."""
        self._thread = threading.Thread(target=self._run, args=(get_settings,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def feed(self, samples, sample_rate):
        """Vom Video/Audio-Loop aufgerufen — MUSS billig bleiben, niemals
        Modell-Code hier. Alle Fehler werden verschluckt: Audio-Trigger ist
        ein optionales Extra, darf den Aufrufer nie stören."""
        try:
            if samples.size == 0:
                return
            if sample_rate != self.sample_rate:
                ratio = self.sample_rate / float(sample_rate)
                n_out = max(1, int(len(samples) * ratio))
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, n_out),
                    np.arange(len(samples)), samples
                ).astype(np.float32)
            with self._lock:
                self._buffer = np.concatenate([self._buffer, samples])
                max_len = int(self.sample_rate * self.window_sec)
                if len(self._buffer) > max_len:
                    self._buffer = self._buffer[-max_len:]
        except Exception:
            pass

    def is_triggered(self):
        """(bool, label_or_None) — welche Kategorie (falls eine) gerade über
        der Schwelle liegt."""
        return self._triggered, self._triggered_label

    # --- Interner Hintergrund-Thread --------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from transformers import ClapModel, ClapProcessor
            self._model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self._processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self._model.eval()
            self.logger.info(f"🎤 [{self.name}] Audio-Trigger-Modell (CLAP) geladen.")
            return True
        except Exception as e:
            self.logger.warning(
                f"⚠️ [{self.name}] Audio-Trigger-Modell konnte nicht geladen werden "
                f"(pip install transformers nötig?) — Feature bleibt inaktiv: {e}"
            )
            self._load_failed = True
            return False

    def _update_text_embeds(self, categories):
        import torch
        with torch.no_grad():
            inputs = self._processor(text=categories, return_tensors="pt", padding=True)
            embeds = _extract_features(self._model.get_text_features(**inputs))
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        self._text_embeds = embeds
        self._text_labels = list(categories)

    def _run(self, get_settings):
        last_settings_check = 0.0
        enabled = False
        categories = []
        threshold = 0.3
        interval = 2.0

        while not self._stop.is_set():
            now = time.time()

            if now - last_settings_check > 5.0:
                try:
                    s = get_settings() or {}
                    enabled = bool(s.get('AUDIO_TRIGGER_ENABLED', False))
                    new_categories = [c.strip() for c in s.get('AUDIO_TRIGGER_CATEGORIES', []) if c.strip()]
                    threshold = float(s.get('AUDIO_TRIGGER_THRESHOLD', 0.3))
                    interval = float(s.get('AUDIO_TRIGGER_INTERVAL_SEC', 2.0))
                    if enabled and new_categories != categories:
                        categories = new_categories
                        if categories and self._ensure_model():
                            try:
                                self._update_text_embeds(categories)
                            except Exception as e:
                                self.logger.warning(f"⚠️ [{self.name}] Audio-Kategorien konnten nicht geladen werden: {e}")
                except Exception:
                    pass
                last_settings_check = now

            if not enabled or not categories or self._text_embeds is None:
                self._triggered = False
                self._triggered_label = None
                self._stop.wait(interval)
                continue

            with self._lock:
                snapshot = self._buffer.copy()

            if len(snapshot) < int(self.sample_rate * 0.5):  # mind. 0.5s Audio nötig
                self._stop.wait(interval)
                continue

            try:
                import torch
                # Parametername je nach transformers-Version unterschiedlich
                # ("audio" neu, "audios" älter/deprecated) — beide abdecken.
                # Welche Exception die Bibliothek beim falschen Namen wirft, ist
                # nicht garantiert dieselbe über Versionen hinweg -> breit fangen.
                try:
                    inputs = self._processor(audio=snapshot, sampling_rate=self.sample_rate, return_tensors="pt")
                except Exception as e_audio:
                    try:
                        inputs = self._processor(audios=snapshot, sampling_rate=self.sample_rate, return_tensors="pt")
                    except Exception as e_audios:
                        raise RuntimeError(
                            f"audio=-Aufruf: {e_audio} | audios=-Aufruf: {e_audios}"
                        ) from e_audios
                with torch.no_grad():
                    audio_embed = _extract_features(self._model.get_audio_features(**inputs))
                    audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
                    sims = (audio_embed @ self._text_embeds.T).squeeze(0)
                best_idx = int(sims.argmax())
                best_score = float(sims[best_idx])
                if best_score >= threshold:
                    self._triggered = True
                    self._triggered_label = self._text_labels[best_idx]
                    self.logger.info(f"🔊 [{self.name}] Audio-Trigger: '{self._triggered_label}' ({best_score:.2f})")
                else:
                    self._triggered = False
                    self._triggered_label = None
            except Exception as e:
                self.logger.warning(f"⚠️ [{self.name}] Audio-Klassifikation fehlgeschlagen: {e}")
                self._triggered = False
                self._triggered_label = None

            self._stop.wait(interval)
