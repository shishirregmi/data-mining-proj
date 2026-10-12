# Run

## Docker

    docker compose up --build

Generated predictions, the SVMlight model, and the Part 2 comparison are saved in output/.

## Python

    python -m pip install -r requirements.txt
    python main.py

SVMlight binaries must be installed and available on PATH for the Python run. Docker fetches and compiles SVMlight during build for research and education use; do not redistribute the image.
