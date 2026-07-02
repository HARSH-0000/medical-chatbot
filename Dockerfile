FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirement files first to leverage Docker cache
COPY requirements.txt setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
