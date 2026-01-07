FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY uv.lock uv.lock
COPY README.md README.md
WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync
ENTRYPOINT ["uv", "run", "src/my_project/evaluate.py"]