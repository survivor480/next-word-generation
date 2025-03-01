# Use the official TensorFlow GPU image
FROM tensorflow/tensorflow:2.14.0-gpu

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container alive
CMD ["tail", "-f", "/dev/null"]