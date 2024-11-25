# Use an official Python runtime as a base image
FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app runs on
EXPOSE 5000

# Set the command to run the application
CMD ["python", "inference_classifier.py"]
