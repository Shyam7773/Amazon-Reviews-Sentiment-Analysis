FROM python:3.11-slim
WORKDIR /app

# install deps
COPY requirements.txt .
COPY serving/ serving/
COPY models/ models/
RUN pip install --no-cache-dir -r requirements.txt

# copying code + artifacts
COPY serving/ serving/
COPY models/ models/

# expose port for API
EXPOSE 8000

# run server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
