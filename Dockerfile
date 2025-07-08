# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose port 8000 (Railway default)
EXPOSE 8000

# Set environment variable for Flask
ENV PORT=8000

# Start the app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "serve:app"]
