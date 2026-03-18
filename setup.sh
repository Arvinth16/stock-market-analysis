#!/bin/bash
# Setup script for India Stock Market Research Agent
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🇮🇳 India Stock Market Research Agent — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$(dirname "$0")"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 11 ]; then
    echo "  ✗ Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "  ✓ Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "  → Creating virtual environment..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# Activate and install
source venv/bin/activate
echo "  → Installing dependencies..."
pip install -e ".[dev]" --quiet
echo "  ✓ Dependencies installed"

# macOS: check for libomp (needed by XGBoost)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list libomp &>/dev/null; then
        echo "  → Installing libomp (required by XGBoost on macOS)..."
        brew install libomp --quiet
        echo "  ✓ libomp installed"
    else
        echo "  ✓ libomp already installed"
    fi
fi

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ Created .env from .env.example"
else
    echo "  ✓ .env already exists"
fi

# Initialize database
python -m src.core.database
echo "  ✓ Database initialized"

# Run tests
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Running tests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pytest tests/ -v --tb=short

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SETUP COMPLETE"
echo ""
echo "To get started:"
echo "  source venv/bin/activate"
echo "  india-stock pipeline          # Run full pipeline"
echo "  india-stock rank --top 10     # Get rankings"
echo "  india-stock news RELIANCE     # Get news for a stock"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
