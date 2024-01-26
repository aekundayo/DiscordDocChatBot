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
RUN apt-get update 
RUN apt-get install -y --no-install-recommends gcc python3-dev 
RUN pip install --no-cache-dir -r requirements.txt 
RUN apt-get remove -y gcc python3-dev 
RUN apt-get autoremove -y 
RUN rm -rf /var/lib/apt/lists/*



# Switch to the new user
USER appuser

# Run bot.py when the container launches
CMD ["python", "main.py"]
