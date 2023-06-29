# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Create a new user "appuser"
RUN useradd appuser

# Add the current directory contents into the container at /app
COPY main.py requirements.txt /app/
RUN mkdir docs

# Change ownership of /app to appuser
RUN chown -R appuser /app

RUN apt-get update && apt-get install -y gcc python3-dev


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the new user
USER appuser

# Run bot.py when the container launches
CMD ["python", "./main.py"]

