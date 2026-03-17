# U-Net Image Segmentation (PyTorch)

Este proyecto nace de una pregunta bastante simple:
¿cómo hace una computadora para “entender” qué hay dentro de una imagen?

No solo reconocer que hay un objeto, sino saber exactamente **dónde empieza y dónde termina**.

Por ejemplo:
- ¿Dónde está un tumor en una imagen médica?
- ¿Dónde termina una célula y empieza otra?
- ¿Qué píxeles pertenecen a una calle en una imagen satelital?
- ¿Qué parte de una imagen corresponde a un producto en una góndola?

Para responder ese tipo de problemas existe una tarea llamada **segmentación de imágenes**, y uno de los modelos más importantes en la historia de esta área es **U-Net**.

---

## Un poco de historia: ¿qué es U-Net y por qué importa?

U-Net es una arquitectura de red neuronal creada en 2015 por investigadores de la Universidad de Freiburg, pensada originalmente para analizar imágenes médicas.

Hasta ese momento, muchos modelos:
- necesitaban miles de imágenes etiquetadas
- tenían dificultades para lograr precisión pixel por pixel

U-Net introdujo una idea clave:
combinar una red que **entiende el contexto general de la imagen** con otra que permite **recuperar detalle fino**, conectándolas mediante lo que hoy se conoce como *skip connections*.

Esa estructura en forma de “U” permitió lograr resultados sorprendentes incluso con pocos datos, y desde entonces U-Net se convirtió en:
- un estándar en segmentación médica  
- la base de muchísimos modelos modernos  
- una arquitectura fundamental para entender visión por computadora avanzada  

Hoy existen variantes como U-Net++, Attention U-Net, nnU-Net, entre muchas otras.

---

## Objetivo de este proyecto

El objetivo no fue solo “hacer que funcione”, sino construir un proyecto completo que permita:

- Entender cómo funciona U-Net internamente  
- Implementar el modelo desde cero en PyTorch  
- Entrenar y validar con métricas adecuadas  
- Visualizar resultados y curvas de aprendizaje  
- Construir un pipeline reproducible  
- Generar predicciones sobre nuevas imágenes  

Es decir: pasar de la teoría a un sistema funcional de punta a punta.

---

## ¿Qué hace exactamente este proyecto?

Este repositorio implementa un pipeline completo de deep learning para segmentación de imágenes, incluyendo:

- Carga y preprocesamiento de datos  
- Definición del modelo U-Net  
- Entrenamiento y validación  
- Implementación de Dice metric y Dice loss  
- Early stopping  
- Visualización de métricas  
- Pipeline de inferencia para generar predicciones  

El flujo principal del proyecto se encuentra dentro de la carpeta `notebooks`.

---

## Resultados

- Mejor Dice en validación: 0.942  
- Epochs utilizados: 25  
- Batch size: 8  
- Optimizador: Adam  

En términos prácticos, esto significa que el modelo logra identificar correctamente la forma de los objetos en imágenes nuevas con alta precisión.

---

## ¿Por qué este proyecto es útil más allá del experimento?

Porque este tipo de técnicas se utilizan hoy en:
- Medicina (diagnóstico asistido por imágenes)
- Visión artificial en industria
- Análisis satelital
- Retail analytics
- Control de calidad
- Biología computacional
- Automatización visual

Este proyecto es una base sólida sobre la que se pueden construir sistemas reales.

---

## Autor

Bruno Dinello / Carlos Dutra da Silveira / Lorenzo Foderé  
Machine Learning & Data Science  
LinkedIn: https://www.linkedin.com/in/bruno-dinello  
GitHub: https://github.com/brunodinello

