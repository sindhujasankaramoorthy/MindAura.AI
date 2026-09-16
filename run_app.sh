#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "🏥 Launching MedTrust AI Clinical Telehealth Platform"
echo "=========================================================="

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Using Python: $($PYTHON_BIN --version)"
echo "Starting FastAPI server on http://localhost:8000 ..."
echo "API Documentation: http://localhost:8000/docs"
echo "Web Application:   http://localhost:8000"
echo "----------------------------------------------------------"

exec $PYTHON_BIN -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
