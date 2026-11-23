"""Servidor Flask: cámara del PC -> pipeline Espectrazo + CNN -> interfaz HTML."""
import os, time, threading
from collections import deque, Counter
from datetime import datetime
from pathlib import Path
import cv2
from flask import Flask, Response, jsonify, request, render_template, send_from_directory
from pipeline import Segmentador, Espectro, Clasificador, crop_para_cnn, redimensionar

BASE = Path(__file__).parent
REC_DIR = BASE / "recordings"
MODEL_PATH = BASE / "models" / "cnn_posturas.pt"


def hex_a_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


class Motor:
    def __init__(self):
        self.params = dict(trail=15, intensidad=0.9, color="#8b5cf6", fondo="video")
        self.clf = Clasificador(MODEL_PATH)
        self.running = False
        self.error = None
        self.fps, self.label, self.conf = 0.0, None, 0.0
        self.hay_silueta, self.aprendiendo = False, False
        self.jpg, self.fid = None, 0
        self.recording, self.last_rec, self.rec_error = False, None, None
        self._calibrar, self._rec_pedido = False, None
        self.thread = None

    def start(self):
        if self.running:
            return
        self.error, self.running = None, True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        self.jpg = None

    def _loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, _ = cap.read()
        if not ok:
            self.error, self.running = "camara", False
            cap.release()
            return
        seg, esp, hist, writer, i = Segmentador(), Espectro(), deque(maxlen=5), None, 0
        t_prev = time.perf_counter()
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.error = "camara"
                    break
                frame = cv2.flip(redimensionar(frame), 1)
                if self._calibrar:
                    seg.reset(); esp.canvas = None; hist.clear(); self._calibrar = False
                comp, info = seg.procesar(frame)
                self.aprendiendo, self.hay_silueta = seg.aprendiendo, comp is not None

                # CNN cada 3 cuadros, con voto de los últimos 5
                if comp is None:
                    hist.clear(); self.label = None
                elif self.clf.ok and i % 3 == 0:
                    crop = crop_para_cnn(frame, comp, info, self.clf.size)
                    if crop is not None:
                        etiqueta, self.conf = self.clf.predecir(crop)
                        hist.append(etiqueta)
                        self.label = Counter(hist).most_common(1)[0][0]

                p = dict(self.params)
                out = esp.actualizar(frame, comp, p["trail"], p["intensidad"], hex_a_bgr(p["color"]), p["fondo"])

                # grabación
                if self._rec_pedido == "start" and writer is None:
                    nombre = datetime.now().strftime("espectrazo_%Y%m%d_%H%M%S.mp4")
                    writer = cv2.VideoWriter(str(REC_DIR / nombre), cv2.VideoWriter_fourcc(*"mp4v"),
                                             20, (out.shape[1], out.shape[0]))
                    if writer.isOpened():
                        self.recording, self.rec_error, self.last_rec = True, None, nombre
                    else:
                        writer, self.rec_error = None, "No se pudo guardar el video. Inténtalo de nuevo."
                    self._rec_pedido = None
                elif self._rec_pedido == "stop":
                    if writer: writer.release(); writer = None
                    self.recording, self._rec_pedido = False, None
                if writer: writer.write(out)

                _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.jpg, self.fid = buf.tobytes(), self.fid + 1
                i += 1
                t = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(t - t_prev, 1e-6) if self.fps else 1 / max(t - t_prev, 1e-6)
                t_prev = t
        finally:
            cap.release()
            if writer: writer.release()
            self.recording, self.running = False, False


motor = Motor()
app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    def gen():
        ultimo = -1
        while motor.running:
            if motor.jpg is not None and motor.fid != ultimo:
                ultimo = motor.fid
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + motor.jpg + b"\r\n"
            else:
                time.sleep(0.01)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/camera")
def camera():
    if (request.json or {}).get("action") == "start":
        motor.start()
    else:
        motor.stop()
    return jsonify(ok=True)


@app.post("/api/params")
def params():
    d = request.json or {}
    if "trail" in d: motor.params["trail"] = max(2, min(60, int(d["trail"])))
    if "intensidad" in d: motor.params["intensidad"] = max(0.1, min(1.0, float(d["intensidad"])))
    if "color" in d and len(d["color"]) == 7: motor.params["color"] = d["color"]
    if d.get("fondo") in ("video", "negro"): motor.params["fondo"] = d["fondo"]
    return jsonify(ok=True)


@app.post("/api/calibrate")
def calibrate():
    motor._calibrar = True
    return jsonify(ok=True)


@app.post("/api/record")
def record():
    motor._rec_pedido = "start" if (request.json or {}).get("action") == "start" else "stop"
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    nombre = motor.label.replace("_", " ").capitalize() if motor.label else None
    return jsonify(running=motor.running, error=motor.error, fps=round(motor.fps, 1),
                   aprendiendo=motor.aprendiendo, hay_silueta=motor.hay_silueta,
                   postura=nombre, confianza=round(motor.conf * 100), modelo=motor.clf.ok,
                   grabando=motor.recording, video=motor.last_rec, error_rec=motor.rec_error)


@app.get("/recordings/<path:f>")
def descargar(f):
    return send_from_directory(REC_DIR, f, as_attachment=True)


if __name__ == "__main__":
    print("Abre http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)
