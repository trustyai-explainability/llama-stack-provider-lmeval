#!/bin/bash
set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Error handler
error_handler() {
    echo -e "${RED}Error on line $1${NC}"
    exit 1
}
trap 'error_handler $LINENO' ERR

# Configuration
IMAGE_NAME="${IMAGE_NAME:-llama-stack-lmeval}"
VERSION="${VERSION:-local}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"

echo -e "${GREEN}🚀 Building ${FULL_IMAGE_NAME} Docker image...${NC}"

# Build the Docker image
docker build -f Dockerfile.local -t "${FULL_IMAGE_NAME}" .

echo -e "${GREEN}✅ Build completed successfully!${NC}"

# Test the installation - Override entrypoint to run python directly
echo -e "${YELLOW}🧪 Testing Python imports...${NC}"
docker run --rm --entrypoint python "${FULL_IMAGE_NAME}" -c "
import sys
sys.path.insert(0, '/app/src')
from llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig
from llama_stack_provider_lmeval.lmeval import LMEval
print('✅ All imports successful!')
"

# Run the unit tests - Override entrypoint
echo -e "${YELLOW}🧪 Running unit tests...${NC}"
docker run --rm \
  --entrypoint python \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/tests:/app/tests:ro" \
  -v "$(pwd)/pytest.ini:/app/pytest.ini:ro" \
  -e PYTHONPATH=/app/src \
  "${FULL_IMAGE_NAME}" \
  -m pytest tests/ -v

echo -e "${GREEN}🎉 All tests passed!${NC}"

# Test the server help command (with default entrypoint)
echo -e "${YELLOW}🔧 Testing server startup (help command)...${NC}"
docker run --rm "${FULL_IMAGE_NAME}" --help

# Display usage information
cat << EOF

${GREEN}📋 Usage Examples:${NC}
===================

1. Run the server with your config:
   docker run -p 8321:8321 -v \$(pwd)/run.yaml:/app/run.yaml:ro \\
     -v \$(pwd)/providers.d:/app/providers.d:ro \\
     ${FULL_IMAGE_NAME} --config /app/run.yaml

2. Run with docker-compose:
   docker-compose up llama-stack-lmeval

3. Run tests:
   docker-compose --profile test up test

4. Development shell:
   docker-compose --profile dev run --rm dev

${GREEN}✨ Your Docker image is ready: ${FULL_IMAGE_NAME}${NC}
EOF