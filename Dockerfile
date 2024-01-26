# Use an official Python runtime as a parent image
FROM python:3.9.18-bullseye as builder



# Set the working directory in the container to /app
WORKDIR /app

# Create a new user "appuser"
RUN useradd appuser

# Add the current directory contents into the container at /app
COPY ./src .
COPY requirements.txt .
RUN mkdir docs


# Change ownership of /app to appuser
RUN chown -R appuser /app


# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && pip install --no-cache-dir -r requirements.txt && apt-get remove -y gcc python3-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* 


# Final stage
FROM python:3.9-slim-bullseye
WORKDIR /app
COPY --from=builder /app /app
RUN useradd appuser && chown -R appuser /app
USER appuser
CMD ["python", "main.py"]