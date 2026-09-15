# Use a small official Python 3.12 image as the program's foundation.
FROM python:3.12-slim

# Prevent Python from creating unnecessary bytecode cache files.
ENV PYTHONDONTWRITEBYTECODE=1
# Send printed results directly to the terminal without buffering.
ENV PYTHONUNBUFFERED=1

# Use /app as the working folder inside the container.
WORKDIR /app

# Copy the dependency list first so Docker can cache installed packages.
COPY requirements.txt ./
# Install the required packages without retaining pip's download cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the classifier source code into the container.
COPY main.py ./
# Copy the supplied training workbook into the container.
COPY ["Training dataset.xlsx", "./"]
# Copy the supplied testing workbook into the container.
COPY ["Testing dataset.xlsx", "./"]

# Run the complete classification experiment when the container starts.
CMD ["python", "main.py"]
