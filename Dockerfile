FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    pkg-config \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Make the source script executable
RUN chmod +x /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/app/src"
ENV MYSQL_USER=$MYSQL_USER
ENV MYSQL_PASSWORD=$PASSWORD_MYSQL

# Install the project in editable mode
RUN pip install -e .

# Expose the port your Streamlit app runs on
EXPOSE 8081

# Set the working directory
WORKDIR /app

# Command to run your Streamlit app
CMD ["streamlit", "run", "main.py"]
