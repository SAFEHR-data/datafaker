FROM python:3.13.3-alpine3.22
RUN apk add bash uv build-base
ENV SHELL=/bin/bash
ENV HOME=/
WORKDIR /app
ADD pyproject.toml uv.lock /app/
RUN uv venv
RUN uv sync --no-install-project
COPY . /app/
RUN uv sync
SHELL ["/bin/bash", "-c"]
# The redirect to /dev/null seems to help shellingham detect bash!
RUN uv run datafaker --install-completion > /dev/null
WORKDIR /data
CMD ["bash", "-c", "source /app/.venv/bin/activate;bash"]
