# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the current directory into the container
COPY . .

# Command to run the FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
