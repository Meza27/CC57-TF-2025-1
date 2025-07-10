# 💹 Crypto Opportunity Predictor - Trabajo Final CC57

Este repositorio contiene el desarrollo completo del trabajo final del curso **CC57 - Machine Learning (2025-I)**, cuyo objetivo fue construir un sistema con modelo de inteligencia artificial capaz de identificar oportunidades de inversión en criptomonedas de baja capitalización, alineado con las narrativas de interés del cliente **Perú C-Inversiones**.

## 📌 Objetivo del Proyecto

Desarrollar un modelo de Machine Learning que prediga el crecimiento futuro de criptomonedas emergentes y lo comunique a través de una interfaz web amigable, permitiendo:

- Predecir el cambio de precio futuro de un criptoactivo.
- Clasificarlo como una oportunidad de inversión (alta, moderada, baja o no recomendada).
- Generar recomendaciones personalizadas según el perfil de riesgo del usuario.
- Sugerir un portafolio diversificado de inversión.

---

## 🧠 Tecnologías y Herramientas

- **Python 3.10+**
- **Scikit-learn**, **PyTorch**, **Pandas**, **Seaborn**, **Matplotlib**
- **Flask** (para la interfaz web)
- **CoinGecko API** (para adquisición de datos en tiempo real)
- **GitHub + Colab** (para desarrollo y documentación)

---

## 📊 Dataset

- Fuente: CoinGecko API
- Se recolectaron más de 5000 criptomonedas y se filtraron por:
  - Narrativas: IA, Videojuegos, RWA, Memes
  - Baja capitalización (volumen < 10M USD)
- Se creó una variable objetivo sintética `cambio_precio_futuro` basada en indicadores técnicos y momentum.

---

## ⚙️ Modelos Implementados

- **Random Forest Regressor** (modelo principal)
- **MLP (Red Neuronal Multicapa)** con PyTorch
- **Autoencoder + KMeans** para detección de patrones

Cada modelo fue evaluado con métricas de regresión y clasificación binaria (conversión basada en umbral del 5%).

---

## 🧪 Resultados

- **Random Forest** obtuvo:
  - AUC-ROC: 0.9822
  - F1-Score: > 90%
  - Retorno simulado: +11,000%
- Se validó con:
  - Validación cruzada (k=5)
  - Winsorización de outliers
  - Evaluación con múltiples umbrales

---

## 🖥️ Interfaz Web

La aplicación web permite:

- Ingresar una criptomoneda y obtener su predicción.
- Generar recomendaciones según perfil de riesgo.
- Obtener un portafolio sugerido con distribución de inversión.

