# Spotify Music Analysis Platform - Makefile
# ==========================================
# 
# Development automation and deployment commands
# 
# Author: Data Science Team
# Version: 1.0.0

.PHONY: help install dev test lint format clean build deploy docker-build docker-run docs

# Default target
help:
	@echo "Spotify Music Analysis Platform - Development Commands"
	@echo "======================================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install        Install dependencies and setup environment"
	@echo "  dev            Setup development environment"
	@echo "  install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run            Run the Flask application"
	@echo "  run-dev        Run in development mode with auto-reload"
	@echo "  run-dashboard  Run Streamlit dashboard"
	@echo "  jupyter        Start Jupyter notebook server"
	@echo ""
	@echo "Testing and Quality:"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo "  security       Run security checks"
	@echo ""
	@echo "Docker and Deployment:"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-run     Run with Docker Compose"
	@echo "  docker-dev     Run development environment with Docker"
	@echo "  docker-prod    Run production environment with Docker"
	@echo "  docker-stop    Stop all Docker services"
	@echo "  docker-clean   Clean Docker resources"
	@echo ""
	@echo "Documentation and Maintenance:"
	@echo "  docs           Build documentation"
	@echo "  docs-serve     Serve documentation locally"
	@echo "  clean          Clean build artifacts and cache"
	@echo "  clean-data     Clean generated data files"
	@echo "  backup         Backup important data"

# Variables
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := spotify-analysis

# Setup and Installation
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('vader_lexicon')"
	@echo "Creating necessary directories..."
	mkdir -p data/uploads data/processed static/images output logs
	@echo "Installation complete!"

dev: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Development environment ready!"

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy pre-commit sphinx jupyter streamlit

# Development Commands
run:
	@echo "Starting Flask application..."
	$(PYTHON) app.py

run-dev:
	@echo "Starting development server..."
	export FLASK_ENV=development && export FLASK_DEBUG=1 && $(PYTHON) app.py

run-dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run dashboard.py

jupyter:
	@echo "Starting Jupyter notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Testing and Quality
test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=spotify_analysis --cov-report=html --cov-report=term

lint:
	@echo "Running linting checks..."
	flake8 spotify_analysis/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	black --check spotify_analysis/ tests/
	isort --check-only spotify_analysis/ tests/

format:
	@echo "Formatting code..."
	black spotify_analysis/ tests/ app.py
	isort spotify_analysis/ tests/ app.py
	@echo "Code formatted successfully!"

type-check:
	@echo "Running type checks..."
	mypy spotify_analysis/ --ignore-missing-imports

security:
	@echo "Running security checks..."
	bandit -r spotify_analysis/ -f json -o security_report.json
	safety check

# Docker Commands
docker-build:
	@echo "Building Docker images..."
	$(DOCKER_COMPOSE) build

docker-run:
	@echo "Starting services with Docker Compose..."
	$(DOCKER_COMPOSE) up -d

docker-dev:
	@echo "Starting development environment..."
	export BUILD_TARGET=development && $(DOCKER_COMPOSE) --profile development up -d

docker-prod:
	@echo "Starting production environment..."
	export BUILD_TARGET=production && $(DOCKER_COMPOSE) up -d

docker-stop:
	@echo "Stopping Docker services..."
	$(DOCKER_COMPOSE) down

docker-clean:
	@echo "Cleaning Docker resources..."
	$(DOCKER_COMPOSE) down -v --rmi all --remove-orphans
	docker system prune -f

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html

docs-serve:
	@echo "Serving documentation locally..."
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Maintenance
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .tox/

clean-data:
	@echo "Cleaning generated data files..."
	rm -rf data/processed/* output/* static/images/*.png
	find logs/ -name "*.log" -type f -delete

backup:
	@echo "Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='*.pyc' \
		--exclude='__pycache__' \
		--exclude='.git' \
		--exclude='data/uploads' \
		--exclude='logs' \
		.

# Database Management
db-init:
	@echo "Initializing database..."
	$(PYTHON) -c "from app import db; db.create_all()"

db-migrate:
	@echo "Running database migrations..."
	flask db upgrade

db-reset:
	@echo "Resetting database..."
	$(PYTHON) -c "from app import db; db.drop_all(); db.create_all()"

# Performance Testing
perf-test:
	@echo "Running performance tests..."
	locust -f tests/performance/locustfile.py --host=http://localhost:5000

load-test:
	@echo "Running load tests..."
	pytest tests/performance/ -v

# Release Management
bump-version:
	@echo "Bumping version..."
	bump2version patch

release:
	@echo "Creating release..."
	git tag -a v$(shell grep __version__ spotify_analysis/__init__.py | cut -d'"' -f2) -m "Release version $(shell grep __version__ spotify_analysis/__init__.py | cut -d'"' -f2)"
	git push origin --tags

# CI/CD
ci-install:
	@echo "CI: Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy

ci-test:
	@echo "CI: Running tests..."
	pytest tests/ -v --cov=spotify_analysis --cov-report=xml

ci-lint:
	@echo "CI: Running linting..."
	black --check spotify_analysis/ tests/
	flake8 spotify_analysis/ tests/ --max-line-length=88
	mypy spotify_analysis/ --ignore-missing-imports

# Monitoring
logs:
	@echo "Showing application logs..."
	tail -f logs/spotify_analysis.log

logs-docker:
	@echo "Showing Docker logs..."
	$(DOCKER_COMPOSE) logs -f spotify-analysis

monitor:
	@echo "Starting monitoring dashboard..."
	$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

# Data Management
sample-data:
	@echo "Downloading sample data..."
	curl -o data/sample/spotify_features_sample.csv \
		"https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"

process-data:
	@echo "Processing sample data..."
	$(PYTHON) scripts/process_sample_data.py

# Environment Management
env-check:
	@echo "Checking environment..."
	$(PYTHON) --version
	$(PIP) --version
	$(PYTHON) -c "import sys; print(f'Python path: {sys.executable}')"
	$(PYTHON) -c "import pandas, numpy, sklearn, flask; print('Core dependencies OK')"

env-info:
	@echo "Environment Information:"
	@echo "======================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Git Branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git Commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repository')"

# Quick Start
quickstart: install sample-data
	@echo "Quick start setup complete!"
	@echo "Run 'make run-dev' to start the development server"
	@echo "Or 'make docker-dev' to start with Docker"

# All-in-one commands
setup: install dev sample-data
	@echo "Complete setup finished!"

deploy-local: docker-build docker-prod
	@echo "Local deployment complete!"
	@echo "Application: http://localhost"
	@echo "API: http://localhost/api"

# Help for specific categories
help-docker:
	@echo "Docker Commands:"
	@echo "==============="
	@echo "  docker-build   Build all Docker images"
	@echo "  docker-dev     Start development environment"
	@echo "  docker-prod    Start production environment"
	@echo "  docker-stop    Stop all services"
	@echo "  docker-clean   Clean all Docker resources"

help-test:
	@echo "Testing Commands:"
	@echo "================"
	@echo "  test          Run unit tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Check code style"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  security      Run security scans"
