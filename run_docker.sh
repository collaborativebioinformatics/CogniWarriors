#!/bin/bash
# Run script for Docker-based federated learning

echo "=== Docker-based Federated Learning with Mixed-Effects Models ==="
echo ""

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Build and start containers
echo "Building and starting containers..."
docker-compose up --build

echo ""
echo "=== Training Complete ==="
echo ""
echo "To view logs:"
echo "  docker-compose logs hub"
echo "  docker-compose logs worker1"
echo "  docker-compose logs worker2"
echo ""
echo "To stop containers:"
echo "  docker-compose down"
echo ""
echo "To remove volumes:"
echo "  docker-compose down -v"