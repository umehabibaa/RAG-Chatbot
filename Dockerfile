# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /code

# 3. Install system dependencies (CRITICAL for ChromaDB/PDF processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Create a user to avoid permission issues
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 6. Copy the rest of your app (including data/ and chroma_db/)
COPY --chown=user . $HOME/app

# 7. Expose the Hugging Face default port
EXPOSE 7860

# 8. Start Streamlit (Ensure it listens on 7860 and 0.0.0.0)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]