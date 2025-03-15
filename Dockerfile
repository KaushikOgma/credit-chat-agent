# Use the official Python base image
FROM python:3.12-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libdbus-1-dev \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app


# Install pip and setuptools in their latest versions
RUN pip install --no-cache-dir --upgrade pip setuptools wheel


# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the FastAPI default port
EXPOSE $APP_PORT

CMD ["python", "main.py"]