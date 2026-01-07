FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY configs/ configs/
COPY README.md README.md
WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync
RUN uv run python src/my_project/data.py data/raw data/processed
ENTRYPOINT ["uv", "run", "src/my_project/train.py"]