# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements first (for layer caching)
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user (recommended for HF Spaces)
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user's home
WORKDIR $HOME/app

# Copy the app code
COPY --chown=user . $HOME/app

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Command to run FastAPI using uvicorn on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
