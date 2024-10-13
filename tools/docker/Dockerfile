# Use the official Python 3.13-rc image as the base image
FROM python:3.12

# Install Poetry, a dependency management tool for Python
RUN pip install poetry

# [Optional] Configure Poetry to create virtual environments inside the project folder
RUN poetry config virtualenvs.in-project false

# Copy the dependency files (pyproject.toml and poetry.lock) to the container
COPY ./pyproject.toml ./poetry.lock ./


# Configure Poetry to install dependencies into the base environment (no virtualenv)
RUN poetry config virtualenvs.create false

# Install project dependencies specified in pyproject.toml
# --no-ansi: Disables colored output
# --no-interaction: Skips any interactive prompts
# --no-root: Does not install the current project as a dependency
RUN poetry install --no-ansi --no-interaction --no-root

# Update the package list
RUN apt-get update 
