# vartificial: Visión Artificial para el Posgrado

Este repositorio contiene los laboratorios prácticos desarrollados para el módulo de **Visión Artificial**. El objetivo principal es comprender cómo las computadoras "ven" y procesan imágenes, desde los fundamentos teóricos hasta el entrenamiento de redes neuronales profundas.

---

## 📋 Módulos y Prácticas

A continuación, se detalla el contenido de los archivos y laboratorios del repositorio:

### 1. Fundamentos de Visión de Imágenes (Scripts Paso a Paso)
Serie de scripts secuenciales diseñados para comprender la manipulación digital de imágenes desde los conceptos base:

*   **`paso1.py`**: Verificación de importación de OpenCV.
*   **`paso2.py`**: Lectura y visualización básica de una imagen (`posgrado.png`) en una ventana nativa utilizando OpenCV.
*   **`paso3.py`**: Carga de imágenes (`boliviaok.png`) con `PIL`, transformación a vectores de `numpy` y análisis a nivel de píxel (recorrido completo de la matriz BGR/RGB).
*   **`paso4.py`**: Conversión del espacio de color BGR a RGB con `cv2.cvtColor` para mostrar imágenes cargadas en OpenCV utilizando `matplotlib`.
*   **`paso5.py`**: Descomposición de canales de color (`R, G, B = cv2.split`) y graficado de un canal específico en escala de grises.
*   **`paso6.py`**: Análisis comparativo visual de canales individuales (Rojo, Verde, Azul) mediante subplots (`plt.subplot`) side-by-side.

**Ejecución:**
```bash
python paso[Numero].py
```

---

### 2. `inicio.py`
**Tema:** Exploración y Visualización Dinámica de MNIST

Este script de introducción sirve para familiarizarse con bases de datos de deep learning mediante la carga directa del dataset MNIST:

*   **Bypass de Certificados:** Cuenta con control y omisión de errores SSL nativo de macOS.
*   **Visualización Dinámica:** Selecciona un índice aleatorio en cada ejecución (`random.randint`) sobre las 60,000 imágenes de entrenamiento y las visualiza en escala de grises con su respectiva etiqueta en Matplotlib.

**Ejecución:**
```bash
python inicio.py
```

---

### 3. `reconoce.py`
**Tema:** Diseño, Entrenamiento y Evaluación de una Red Neuronal Convolucional (CNN)

Este script implementa el flujo de trabajo clásico de Machine Learning utilizando el dataset **MNIST** para el reconocimiento de dígitos escritos a mano:

*   **Preprocesamiento:** Carga el dataset, realiza la normalización de píxeles (0.0 a 1.0) y redimensiona las imágenes a la forma convolucional `(ancho, alto, canal)`.
*   **Arquitectura CNN:** Construye una red neuronal convolucional robusta compuesta por 3 capas `Conv2D` con activación `relu`, capas `MaxPooling2D` para reducción espacial, una capa de aplanado (`Flatten`), una capa totalmente conectada (`Dense` de 64 neuronas) y una capa de salida `Softmax` de 10 neuronas.
*   **Entrenamiento:** Compila con optimizador Adam y pérdida de entropía cruzada categórica escasa, entrenando el modelo por 5 épocas.
*   **Evaluación y Simulación:** Evalúa la precisión sobre datos de prueba (MNIST Test) y simula una predicción real seleccionando una imagen aleatoria, mostrando en consola la matriz de probabilidades de salida junto a la gráfica del dígito analizado.

**Ejecución:**
```bash
python reconoce.py
```

---

### 4. `leeReconoce.py`
**Tema:** Clasificación de Dígitos por Cámara en Tiempo Real

Este es el proyecto integrador del módulo de visión artificial, el cual combina el entrenamiento de la red neuronal convolucional (CNN) con la captura de video en vivo:

*   **Entrenamiento Rápido:** Entrena un modelo CNN idéntico con MNIST (en 3 épocas para velocidad) al iniciar la ejecución del programa.
*   **Visión en Vivo:** Inicializa la cámara web mediante OpenCV (`cv2.VideoCapture`).
*   **Segmentación Digital (ROI):** Define un cuadro guía (verde) en el centro de la pantalla. El usuario coloca el número escrito en un papel dentro del recuadro.
*   **Binarización e Inversión:** Al pulsar la **barra espaciadora**, el programa recorta la Zona de Interés (ROI), la convierte a escala de grises, la binariza con umbralización inversa (para que el fondo sea negro y el trazo blanco como en MNIST) y la escala a 28x28 píxeles.
*   **Predicción Dinámica:** Ejecuta la predicción del modelo y muestra en consola la certeza obtenida (porcentaje de probabilidad).

**Instrucciones de Uso:**
1.  Inicie el programa.
2.  Coloque un número oscuro e independiente dibujado con trazo grueso sobre fondo blanco dentro del recuadro verde.
3.  Presione la **Barra Espaciadora** para evaluar la predicción.
4.  Presione **'q'** para salir del programa y liberar la cámara.

**Ejecución:**
```bash
python leeReconoce.py
```

---

### 5. `fashion.py`
**Tema:** Clasificación de Ropa con Fashion MNIST

Este script utiliza el dataset **Fashion MNIST** (que contiene 70,000 imágenes en escala de grises de 28x28 píxeles divididas en 10 categorías de prendas de vestir) para entrenar una Red Neuronal Convolucional (CNN):

*   **Preprocesamiento:** Carga y normaliza el dataset dividiéndolo en 255.
*   **Arquitectura CNN:** Construye una red secuencial utilizando una capa `Input` para especificar las dimensiones, una capa convolucional `Conv2D` con 16 filtros, una capa `MaxPooling2D` para reducir dimensiones, una capa de aplanado (`Flatten`), y una capa final `Dense` de salida con activación `softmax` para clasificar las 10 categorías de ropa.
*   **Determinismo:** Configura semillas aleatorias fijas (`random`, `numpy` y `tensorflow`) y activa `enable_op_determinism` para garantizar la reproducibilidad de los resultados.
*   **Entrenamiento:** Compila con el optimizador Adam y la pérdida de entropía cruzada categórica escasa (`sparse_categorical_crossentropy`), entrenando por 10 épocas con un tamaño de lote de 256.
*   **Predicción:** Realiza y muestra en consola la predicción sobre una muestra de prueba seleccionada (índice 3322) comparándola con su etiqueta real.

**Ejecución:**
```bash
python fashion.py
```

---

### 6. `yolo.py`
**Tema:** Detección de Objetos en Tiempo Real, Videos y URLs con YOLOv11

Este script implementa la detección de objetos utilizando la arquitectura **YOLOv11** (`yolo11n.pt`) preentrenada:

*   **Soporte Multifuente:** Admite imágenes estáticas locales, videos y URLs de streaming (por ejemplo, videos o transmisiones cortas de YouTube).
*   **Procesamiento por Lotes y Streams:** Configurado con `stream=True` para procesar flujos de video fotograma a fotograma de manera eficiente.
*   **Guardado Automático:** Crea un directorio `resultados` y guarda las imágenes procesadas. En el caso de transmisiones, los fotogramas son etiquetados secuencialmente (`frame_0001.jpg`, `frame_0002.jpg`, etc.).
*   **Reporte de Detecciones:** Imprime directamente en la terminal el resumen de los objetos detectados (como personas, vehículos, mochilas, etc.) junto con sus respectivos porcentajes de confianza.

**Ejecución:**
```bash
python yolo.py
```

---

### 7. `reconoceManosArchivo.py`
**Tema:** Detección de Puntos Clave de la Mano en Imágenes Estáticas (MediaPipe Tasks API)

Este script usa la **nueva API de MediaPipe Tasks** para procesar una imagen estática y dibujar el esqueleto de la mano:

*   **Nueva Tasks API:** Carga un modelo externo `hand_landmarker.task` (igual que YOLO carga `yolo11n.pt`), eliminando por completo los warnings de `deprecated`.
*   **Dibujo Manual de Articulaciones:** Usa `cv2.circle` y `cv2.line` con la lista `HAND_CONNECTIONS` definida en el propio script para trazar los 21 puntos y sus conexiones óseas.
*   **Visualización y Guardado:** Muestra el resultado en pantalla con `cv2.imshow` e imprime en consola las coordenadas en píxeles de la muñeca. Guarda la imagen anotada en `resultados/`.

**Ejecución:**
```bash
python reconoceManosArchivo.py
```

---

### 8. `reconoceManosCamara.py`
**Tema:** Detección de Puntos Clave de la Mano en Tiempo Real (MediaPipe Tasks API)

Este script utiliza la **nueva API de MediaPipe Tasks** en modo video para detectar y trazar manos en tiempo real con la cámara web:

*   **Modo `RunningMode.VIDEO`:** Optimizado para flujos continuos: detecta la mano la primera vez y luego la rastrea fotograma a fotograma, siendo más rápido que redetectar en cada cuadro.
*   **Timestamps Incrementales:** La nueva API requiere pasar el tiempo en milisegundos de cada fotograma para mantener la coherencia del rastreo.
*   **Efecto Espejo (`cv2.flip`):** Voltea la imagen horizontalmente para una visualización más natural.
*   **Superposición de Texto:** Muestra en pantalla la cantidad de manos detectadas. Presiona **'q'** para salir.

**Ejecución:**
```bash
python reconoceManosCamara.py
```

---

## 🛠️ Instalación y Requisitos

Para ejecutar estos programas correctamente, necesita tener instalado Python y algunas librerías específicas.

1.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv .venv
    ```

2.  **Activar el Entorno Virtual:**
    -   **Windows:** `.\.venv\Scripts\activate`
    -   **macOS/Linux:** `source .venv/bin/activate`

3.  **Instalar Dependencias:**
    ```bash
    pip install opencv-python numpy tensorflow matplotlib pillow ultralytics imutils mediapipe
    ```

4.  **Descargar el Modelo de Manos (MediaPipe Tasks API):**
    ```bash
    mkdir -p modelos
    curl -L -o modelos/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    ```

---

## 📂 Estructura del Repositorio

```
vartificial/
├── .venv/                 # Entorno virtual de Python
├── paso1.py a paso6.py    # Laboratorios secuenciales: Fundamentos de OpenCV
├── inicio.py              # Visualización dinámica y aleatoria del dataset MNIST
├── reconoce.py            # Laboratorio CNN: Entrenamiento y validación estática
├── leeReconoce.py         # Laboratorio Final: Reconocimiento y cámara en tiempo real
├── fashion.py             # Laboratorio CNN: Clasificación con Fashion MNIST
├── yolo.py                 # Laboratorio YOLO: Detección de objetos multi-fuente
├── reconoceManosArchivo.py  # Laboratorio MediaPipe Tasks: Detección de manos en imagen
├── reconoceManosCamara.py   # Laboratorio MediaPipe Tasks: Detección de manos en vivo
├── modelos/                 # Modelos de IA externos (.task, .pt)
├── resultados/              # Imágenes anotadas generadas por YOLO y MediaPipe
├── img/                     # Imágenes de entrada para los laboratorios
├── *.png                    # Imágenes adicionales del repositorio
└── README.md                # Este archivo descriptivo del proyecto
```
