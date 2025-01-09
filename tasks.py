import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_individual"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS, in_stream=False
    )  # needed add in_stream False to remove OSError Errno 25 https://github.com/fabric/fabric/issues/2129
    print("hi3")


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/train.py --lr 1e-4 --batch-size 32 --epochs 3",
        echo=True,
        pty=not WINDOWS,
        in_stream=False,
    )


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(
        f"python src/{PROJECT_NAME}/evaluate.py --model-checkpoint models/model.pth",
        echo=True,
        pty=not WINDOWS,
        in_stream=False,
    )


@task
def visualize(ctx: Context) -> None:
    """Visualize tSNE of layer outputs before classification head for test-set."""
    ctx.run(
        f"python src/{PROJECT_NAME}/visualize.py --model-checkpoint models/model.pth --figure-name tSNE.png",
        echo=True,
        pty=not WINDOWS,
        in_stream=False,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS, in_stream=False)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS, in_stream=False)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
