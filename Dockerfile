FROM python:3.9-slim

# Maintainer info
LABEL maintainer="teddyantonius7@gmail.com"

# Make working directories
RUN mkdir -p /wander-recommender-api
WORKDIR /wander-recommender-api

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY . .

# Expose the port the app runs on
EXPOSE 5050

# Run the python application using uvicorn
CMD ["python", "main.py"]
