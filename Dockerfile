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
EXPOSE 8080


# Define environment variable
ENV PORT 8080

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]