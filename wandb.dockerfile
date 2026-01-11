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
RUN uv add wandb
COPY wandb_tester.py wandb_tester.py
ENTRYPOINT ["uv", "run", "wandb_tester.py"]