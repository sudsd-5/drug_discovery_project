.PHONY: help setup install clean test format lint run train predict

help:
	@echo "Drug Discovery Platform - Available Commands"
	@echo "=============================================="
	@echo "setup      - Set up development environment"
	@echo "install    - Install dependencies"
	@echo "clean      - Clean temporary files and caches"
	@echo "test       - Run tests"
	@echo "format     - Format code with black and isort"
	@echo "lint       - Run linting with flake8"
	@echo "run        - Run the complete pipeline"
	@echo "train      - Train the model"
	@echo "predict    - Run prediction (requires MODEL_PATH)"
	@echo "tensorboard - Start TensorBoard"

setup:
	@echo "Setting up development environment..."
	bash scripts/setup_env.sh

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	@echo "Clean complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=. --cov-report=term-missing

format:
	@echo "Formatting code..."
	isort .
	black .
	@echo "Code formatted!"

lint:
	@echo "Running linting..."
	flake8 .

run:
	@echo "Running complete pipeline..."
	python start.py

train:
	@echo "Training model..."
	python train_dti.py

predict:
	@echo "Running prediction..."
	python predict_dti.py

tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir runs --port 6006

# Development shortcuts
dev-setup: setup install
	@echo "Development environment ready!"

dev-check: format lint test
	@echo "All checks passed!"

# Quick start for new users
quickstart:
	@echo "Quick start guide:"
	@echo "1. Run 'make setup' to set up environment"
	@echo "2. Place data files in data/raw/"
	@echo "3. Run 'make run' to execute the pipeline"
	@echo "4. Use 'make tensorboard' to view training progress"
