# image_processing
Procesamiento de Imágenes con Histogram Equalization y CLAHE

Este proyecto permite mejorar el contraste de imágenes utilizando dos técnicas principales:

    Histogram Equalization (Ecualización de Histograma): Mejora el contraste global redistribuyendo los valores de intensidad.
    CLAHE (Contrast Limited Adaptive Histogram Equalization): Mejora el contraste local dividiendo la imagen en bloques pequeños y ajustando sus valores de intensidad.


¿Como ejecutar este proyecto?

python procesar_imagenes.py <carpeta_entrada> <carpeta_salida> <metodo>

    <carpeta_entrada>: Ruta a la carpeta que contiene las imágenes originales.
    <carpeta_salida>: Ruta a la carpeta donde se guardarán las imágenes procesadas.
    <metodo>: Método de procesamiento a utilizar. Puede ser:
        ecualizacion (Histogram Equalization)
        clahe (CLAHE)

Ejemplo de Ejecución

Si tienes imágenes en la carpeta imagenes/ y deseas guardarlas en resultados/ usando CLAHE, ejecuta:

python procesar_imagenes.py imagenes_clahe/ resultados/ clahe

Para usar Histogram Equalization, el comando sería:

python procesar_imagenes.py imagenes_equalization/ resultados/ ecualizacion

Métodos de Mejora de Contraste
Ecualización de Histograma

Este método redistribuye los valores de intensidad para extender el rango de contraste de la imagen. Es útil para imágenes con poco contraste global.

Ventajas:

    Simple y rápido de aplicar.
    Mejora el contraste global.

Limitaciones:

    Puede no preservar detalles locales.
    Puede generar artefactos en imágenes con variaciones sutiles.

CLAHE

Este método divide la imagen en pequeños bloques y aplica ecualización de histograma a cada uno, ajustando los valores para evitar la amplificación excesiva del contraste.

Ventajas:

    Mejora el contraste local en diferentes partes de la imagen.
    Controla la amplificación del ruido.

Limitaciones:

    Más lento que la ecualización de histograma estándar debido a su procesamiento por bloques.
    Requiere configurar parámetros (e.g., tamaño de los bloques).

Estructura del Proyecto

proyecto_imagenes/
├── images_clahe/               # Carpeta que contiene las imágenes originales para clahe
├── images_equalization/        # Carpeta que contiene las imágenes originales para la equalizacion
├── resultados/                 # Carpeta para guardar imágenes procesadas
├── procesar_imagenes.py        # Script principal
└── README.md                   # Documentación del proyecto



contrast limited adaptive histogram equalization (CLAHE)
prevents this by limiting the amplification
• CLAHE limits the amplification by clipping the histogram
at a predefined value before computing the CDF
