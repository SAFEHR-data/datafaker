FROM python:3.14.6-alpine3.24
RUN apk add bash poetry build-base
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN touch README.md
RUN mkdir /pypoetry
ENV POETRY_VIRTUALENVS_PATH=/pypoetry/cache/virtualenv
ENV SHELL=/bin/bash
ENV HOME=/
RUN poetry install --no-interaction --without dev --no-root
COPY . /app
RUN poetry install --only-root
SHELL ["/bin/bash", "-c"]
# The redirect to /dev/null seems to help shellingham detect bash!
RUN poetry run datafaker --install-completion > /dev/null
WORKDIR /data
CMD ["bash", "-c", "source $(poetry -C /app env info --path)/bin/activate;bash"]
