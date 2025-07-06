FROM python:3.10-slim

# Install system dependencies and Chrome
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    unzip \
    && mkdir -p /usr/share/man/man1 /usr/share/keyrings \
    && curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] http://dl.google.com/linux/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install ChromeDriver v135.0.7049.95 explicitly
RUN wget https://storage.googleapis.com/chrome-for-testing-public/135.0.7049.95/linux64/chromedriver-linux64.zip && \
    unzip chromedriver-linux64.zip && \
    mv chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf chromedriver-linux64.zip chromedriver-linux64

# Set environment variables for Chrome
ENV PATH="/usr/local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p data models outputs
# Copy application code and data folder
COPY satte.py ./
COPY data/ ./data/
# Create required folders


# Entrypoint
ENTRYPOINT ["python", "satte.py"]

# Default CMD argument (optional)
CMD ["1"]
