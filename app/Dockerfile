FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/app/entrypoint.sh"]

COPY . /app

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    chmod +x /app/entrypoint.sh
    
CMD ["/bin/bash"]
