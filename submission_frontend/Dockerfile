FROM python:3.11-slim

WORKDIR /app

# Install dependencies directly
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn>=0.30.0" \
    "google-adk>=2.0.0,<3.0.0" \
    "google-cloud-aiplatform>=1.156.0" \
    "jinja2>=3.1.2"

COPY main.py .

EXPOSE 8080

CMD ["python", "main.py"]
