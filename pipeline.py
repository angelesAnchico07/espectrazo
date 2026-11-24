"""Pipeline de Espectrazo: silueta (MOG2 + morfología), estela y clasificador CNN."""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

# Mismos parámetros que el notebook de Colab (para que la CNN vea lo mismo que en el entrenamiento)
FRAMES_FONDO = 30
UMBRAL_DIF = 35   # diferencia mínima (0-255) respecto al fondo para contar como persona; ignora cambios leves de luz
LR_CON_PERSONA, LR_SIN_PERSONA, LR_FANTASMA = 0.0, 0.005, 0.002   # con persona el fondo no se toca (no la "absorbe"); sin persona se actualiza
AREA_MIN_FRAC, AREA_MAX_FRAC, ALTO_MIN_FRAC = 0.03, 0.45, 0.25


def redimensionar(frame, lado=640):
    h, w = frame.shape[:2]
    esc = lado / max(h, w)
    return cv2.resize(frame, (int(w * esc), int(h * esc)))


class CNN(nn.Module):  # idéntica a la del notebook
    def __init__(self, n_clases, img_size=64, dropout=0.5):
        super().__init__()

        def bloque(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(bloque(3, 16), bloque(16, 32), bloque(32, 64))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(64 * (img_size // 8) ** 2, 128), nn.ReLU(),
            nn.Linear(128, n_clases))

    def forward(self, x):
        return self.classifier(self.features(x))


class Segmentador:
    """Etapas 2-4: filtro gaussiano, MOG2, umbral, apertura/cierre y componente de mayor área."""
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.reset()

    def reset(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
        self.n = 0
        self.hay_persona = False
        self.hay_fantasma = False

    @staticmethod
    def _dif(a, b):
        """Diferencia promedio (0-255) entre dos imágenes BGR, por píxel."""
        return cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY)

    def _compensar_luz(self, blur, bg):
        """La cámara cambia sola la exposición cuando entra la persona y eso aclara/oscurece toda la pared.
        Se calcula la ganancia solo con los píxeles que se parecen al fondo (la persona se excluye porque
        se diferencia mucho de él) y se corrige el cuadro."""
        f, b = blur[::4, ::4], bg[::4, ::4]
        libre = self._dif(f, b) < UMBRAL_DIF
        if libre.mean() < 0.15:                        # cambio de luz muy brusco: usar todos los píxeles
            libre = np.ones_like(libre)
        r = (f[libre].astype(np.float32) + 1) / (b[libre].astype(np.float32) + 1)
        g = np.clip(np.median(r, axis=0), 0.6, 1.6)
        return np.clip(blur.astype(np.float32) / g, 0, 255).astype(np.uint8)

    @staticmethod
    def _bordes(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

    def _es_fantasma(self, comp, e_act, e_bg):
        """Un 'fantasma' es la zona donde estaba la persona cuando se aprendió el fondo: el fondo guardado
        tiene bordes allí (la silueta de la persona) pero la imagen actual NO (es solo pared).
        Un objeto real es al revés: bordes en la imagen actual, no en el fondo guardado."""
        banda = cv2.morphologyEx(comp, cv2.MORPH_GRADIENT, self.kernel) > 0
        if banda.sum() < 20:
            return False
        return e_bg[banda].mean() > 1.3 * e_act[banda].mean()

    @property
    def aprendiendo(self):
        return self.n < FRAMES_FONDO

    def procesar(self, frame):
        """Devuelve (mascara_uint8 | None, (x, y, w, h, frac))."""
        aprendiendo = self.aprendiendo
        self.stats = None
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        if aprendiendo:
            mask = self.mog.apply(blur, learningRate=-1)
            self.n += 1
            return None, None
        bg = self.mog.getBackgroundImage()
        blur = self._compensar_luz(blur, bg)
        lr = LR_FANTASMA if self.hay_fantasma else (LR_CON_PERSONA if self.hay_persona else LR_SIN_PERSONA)
        mask = self.mog.apply(blur, learningRate=lr)
        self.n += 1
        mask[self._dif(blur, bg) < UMBRAL_DIF] = 0     # solo lo que se diferencia MUCHO del fondo es persona
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_cierre, iterations=2)  # tapa huecos
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        self.hay_persona = self.hay_fantasma = False
        if n <= 1:
            return None, None
        H, W = mask.shape
        e_act, e_bg = self._bordes(blur), self._bordes(bg)
        i = None
        for c in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:3]:      # los 3 más grandes
            if stats[c, cv2.CC_STAT_AREA] / (H * W) < AREA_MIN_FRAC:
                break
            if self._es_fantasma((labels == c).astype(np.uint8), e_act, e_bg):
                self.hay_fantasma = True          # se va "olvidando" poco a poco (ver LR_FANTASMA)
            else:
                i = int(c)
                break
        if i is None:
            return None, None
        frac = stats[i, cv2.CC_STAT_AREA] / (H * W)
        self.hay_persona = True
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        comp = (labels == i).astype(np.uint8)
        area_antes = int(comp.sum())
        # Rellenar todos los huecos internos: la silueta queda sólida, sin manchas negras
        contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp = np.zeros_like(comp)
        cv2.drawContours(comp, contornos, -1, 1, thickness=cv2.FILLED)
        # Datos para evaluar.py: componentes espurios (todos menos la silueta) y huecos rellenados
        self.stats = dict(espurios=n - 2, area_antes=area_antes, area_final=int(comp.sum()))
        return comp, (x, y, w, h, frac)


def crop_para_cnn(frame, comp, info, size=64):
    """Mismo recorte que el dataset (persona con su color sobre negro, cuadrado, size x size).
    Devuelve None si no pasa los filtros del dataset."""
    if comp is None:
        return None
    x, y, w, h, frac = info
    H = frame.shape[0]
    if frac > AREA_MAX_FRAC or h < ALTO_MIN_FRAC * H or w / h > 1.8 or h / w > 4:
        return None
    rgb = frame.copy()
    rgb[comp == 0] = 0
    crop = rgb[y:y + h, x:x + w]
    lado = max(w, h)
    lienzo = np.zeros((lado, lado, 3), np.uint8)
    lienzo[(lado - h) // 2:(lado - h) // 2 + h, (lado - w) // 2:(lado - w) // 2 + w] = crop
    return cv2.resize(lienzo, (size, size), interpolation=cv2.INTER_AREA)


class Espectro:
    """Etapas 5-6: mapa de estela con decaimiento sobre el video normal.
    `canvas` guarda la opacidad (0..1) de la estela en cada píxel."""
    def __init__(self):
        self.canvas = None

    def actualizar(self, frame, comp, trail, intensidad, color_bgr, fondo="video"):
        h, w = frame.shape[:2]
        if self.canvas is None or self.canvas.shape != (h, w):
            self.canvas = np.zeros((h, w), np.float32)
        self.canvas *= 0.05 ** (1.0 / max(trail, 1))      # tras `trail` cuadros queda al 5 %
        m = None
        if comp is not None:
            m = comp.astype(bool)
            self.canvas[m] = 1.0                           # donde estás ahora, la estela arranca al 100 %
        a = (self.canvas * intensidad)[..., None]
        color = np.array(color_bgr, np.float32)
        base = frame.astype(np.float32) if fondo == "video" else np.zeros(frame.shape, np.float32)
        out = base * (1 - a) + color * a                   # estela morada mezclada sobre el video
        if m is not None and fondo == "video":
            out[m] = frame[m]                              # tu cuerpo se ve normal; la estela queda detrás
        return np.clip(out, 0, 255).astype(np.uint8)


class Clasificador:
    def __init__(self, ruta):
        self.ok, self.clases, self.size = False, [], 64
        if not Path(ruta).exists():
            return
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]

        # Deducir tamaño y nº de clases de los pesos (más confiable que el metadato)
        n_out = sd["classifier.4.weight"].shape[0]
        in_feat = sd["classifier.2.weight"].shape[1]          # 64 * (size//8)**2
        self.size = int(round(8 * (in_feat / 64) ** 0.5))     # 4096 -> 64

        self.clases = list(ck["clases"])
        # El checkpoint guardó solo 3 nombres, pero se entrenó con 4 (README: agachado, brazos abiertos,
        # brazos arriba, de pie). En orden alfabético (como ImageFolder), 'agachado' va primero.
        if n_out == 4 and len(self.clases) == 3 and "agachado" not in self.clases:
            self.clases = ["agachado"] + self.clases
        if len(self.clases) != n_out:
            print(f"[AVISO] El modelo tiene {n_out} salidas pero 'clases' trae {len(self.clases)}: {self.clases}")
            while len(self.clases) < n_out:
                self.clases.append(f"clase_{len(self.clases)}")

        self.model = CNN(n_out, self.size)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.ok = True

    @torch.no_grad()
    def predecir(self, crop_bgr):
        x = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy((x - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0)
        p = torch.softmax(self.model(x), 1)[0]
        i = int(p.argmax())
        return self.clases[i], float(p[i])