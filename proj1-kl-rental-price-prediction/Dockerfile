FROM python:3.14.3-slim-bookworm

# XGBoost requires OpenMP runtime on slim Debian images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY features.py predict.py model.bin model_metadata.json ./

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
