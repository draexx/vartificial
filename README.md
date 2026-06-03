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
    pip install opencv-python numpy tensorflow matplotlib pillow
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
├── *.png                  # Archivos de imágenes utilizados en los laboratorios
└── README.md              # Este archivo descriptivo del proyecto
```
