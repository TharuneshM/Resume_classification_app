FROM python:3.10.13

# Set up directories
RUN mkdir -p /app/store
RUN mkdir -p /app/static

# Set working directory
WORKDIR /app

# Copy the contents of the current directory into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8501  

# Command to run the Streamlit app
CMD ["streamlit", "run", "app_test.py"]

