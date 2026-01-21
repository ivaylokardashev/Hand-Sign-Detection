# Step 1: Use a professional 'slim' base image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies for OpenCV and MediaPipe
# 'slim' images are tiny, so we must add basic libraries for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy only the requirements first (optimizes Docker caching)
COPY requirements.txt .

# Step 5: Install Python libraries
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy your project files into the container
COPY . .

# Step 7: Expose the port your FastAPI/Flask app will run on
EXPOSE 8000

# Step 8: Define the command to start your AI service
CMD ["python", "app/main.py"]