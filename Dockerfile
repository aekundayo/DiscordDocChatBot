# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster as builder

# Set the working directory in the container to /app
WORKDIR /app

# Create a new user "appuser"
RUN useradd appuser

# Add the current directory contents into the container at /app
COPY main.py requirements.txt /app/
RUN mkdir docs
RUN mkdir docs/web
RUN mkdir docs/pdfs

# Change ownership of /app to appuser
RUN chown -R appuser /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Switch to the new user
USER appuser

# Run bot.py when the container launches
CMD ["python", "main.py"]
