FROM lmsysorg/sglang:qwen38-27b

WORKDIR /opt/blackwell-qwen
COPY . .
RUN chmod +x scripts/*.sh

ENV PROFILE=throughput \
    MODEL_PATH=/model \
    SERVED_MODEL_NAME=Qwen/Qwen3.8-27B-FP8

EXPOSE 8000
ENTRYPOINT ["/opt/blackwell-qwen/scripts/serve.sh"]

