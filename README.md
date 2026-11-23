# ESPECTRAZO

Estela en tiempo real del movimiento corporal (MOG2 + morfología + acumulación con transparencia) y
clasificación de postura con una CNN (agachado, brazos abiertos, brazos arriba, de pie). Interfaz web en HTML.

## 1. Exportar el modelo desde Colab
Al final del notebook, ejecuta (usa `model_base`, que tuvo mejor test: 86,2 %):

```python
torch.save({'state_dict': model_base.state_dict(), 'clases': CLASES, 'img_size': IMG_SIZE},
           '/content/drive/MyDrive/Parcial 2 PDI/cnn_posturas.pt')
```
Descarga `cnn_posturas.pt` y ponlo en `models/`.

## 2. Ejecutar en Visual Studio Code
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Abre http://127.0.0.1:5000, pulsa **Iniciar cámara** y sal del cuadro ~2 s mientras aprende el fondo.

## 3. Subir a GitHub
Crea un repositorio vacío en github.com y, dentro de esta carpeta:
```bash
git init
git add .
git commit -m "Espectrazo: pipeline + CNN + interfaz web"
git branch -M main
git remote add origin https://github.com/USUARIO/espectrazo.git
git push -u origin main
```
Guarda también el notebook en `notebooks/` (en Colab: Archivo → Guardar una copia en GitHub).

## Estructura
- `app.py` servidor Flask, cámara y endpoints · `pipeline.py` silueta, estela y CNN · `templates/index.html` interfaz
