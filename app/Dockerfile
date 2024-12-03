# Basis-Image
FROM python:3.12-slim

# Setzen des Arbeitsverzeichnisses im Container
WORKDIR /app

# Installieren der notwendigen Systempakete für tkinter
RUN apt-get update && apt-get install -y \
    python3-tk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && mkdir /app/input /app/output \
    && apt-get clean

# Kopieren der Python-Dateien und anderer notwendiger Dateien ins Container-Image
COPY . .

# Installieren der Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Exponieren des Ports, den Streamlit verwendet
EXPOSE 8501

# Starten der Streamlit-App
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]