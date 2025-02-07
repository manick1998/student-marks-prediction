# Use official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all necessary files
COPY requirements.txt .
COPY src/ src/
COPY models/ models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI's default port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "src.deployment:app", "--host", "0.0.0.0", "--port", "8000"]
 
