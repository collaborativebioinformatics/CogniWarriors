# Docker-based Federated Learning

This guide explains how to run the mixed-effects federated learning system using Docker containers.

## Prerequisites

- Docker installed
- Docker Compose installed

## Quick Start

### Option 1: Run with Docker Compose (Recommended)

```bash
./run_docker.sh
```

Or manually:
```bash
docker-compose up --build
```

### Option 2: Run individual containers

#### Start Hub Container
```bash
docker build -f Dockerfile.hub -t federated_hub .
docker run -d --name hub -p 8080:8080 --network federated_net federated_hub
```

#### Start Worker Containers
```bash
docker build -f Dockerfile.worker -t federated_worker .

docker run -d --name worker1 --network federated_net federated_worker \
  python3 worker.py --hub-address hub:8080 --worker-id worker_01 --data-dir /data/center01 --n-local-epochs 5 --n-rounds 3

docker run -d --name worker2 --network federated_net federated_worker \
  python3 worker.py --hub-address hub:8080 --worker-id worker_02 --data-dir /data/center02 --n-local-epochs 5 --n-rounds 3
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Hub Container                      │   │
│  │  - Global Model (Fixed Effects)                 │   │
│  │  - FedAvg Aggregator                            │   │
│  │  - Round Manager                                │   │
│  │  - Weight Distributor                           │   │
│  │  - Port: 8080                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         │               │               │               │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐       │
│  │ Worker 01   │ │ Worker 02   │ │ Worker N    │       │
│  │ - Local     │ │ - Local     │ │ - Local     │       │
│  │   Training  │ │   Training  │ │   Training  │       │
│  │ - Random    │ │ - Random    │ │ - Random    │       │
│  │   Effects   │ │   Effects   │ │   Effects   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Hub Configuration
- **Port**: 8080 (exposed to host)
- **Host**: 0.0.0.0 (listens on all interfaces)
- **Workers**: 2 (configurable)
- **Rounds**: 3 (configurable)
- **Local Epochs**: 5 (configurable)

### Worker Configuration
- **Hub Address**: hub:8080 (Docker service name)
- **Worker ID**: worker_01, worker_02 (unique identifiers)
- **Data Directory**: /data/center01, /data/center02
- **Local Epochs**: 5 (configurable)
- **Rounds**: 3 (configurable)

## Networking

### Docker Network
- **Network Name**: federated_net
- **Driver**: bridge
- **Service Discovery**: Workers connect to hub via service name "hub"

### Port Mapping
- **Hub**: 8080:8080 (host:container)
- **Workers**: No port mapping needed (connect to hub via Docker network)

## Data Management

### Volumes
- **hub_data**: Shared hub data
- **worker1_data**: Worker 01 local data
- **worker2_data**: Worker 02 local data

### Data Directories
- Hub: /data (shared)
- Workers: /data/center01, /data/center02 (local to each worker)

## Monitoring

### View Logs
```bash
# All containers
docker-compose logs

# Specific container
docker-compose logs hub
docker-compose logs worker1
docker-compose logs worker2

# Follow logs
docker-compose logs -f hub
```

### Check Container Status
```bash
docker-compose ps
```

### Access Container Shell
```bash
docker-compose exec hub bash
docker-compose exec worker1 bash
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs hub

# Rebuild containers
docker-compose down
docker-compose up --build
```

### Network Issues
```bash
# Check network
docker network ls
docker network inspect longitudinal_imaging_to_multimodality_federated_net

# Test connectivity
docker-compose exec worker1 ping hub
```

### Port Conflicts
```bash
# Check if port 8080 is in use
lsof -i :8080

# Kill existing process
pkill -f "center.py"

# Restart containers
docker-compose restart
```

### Remove Everything
```bash
# Stop and remove containers, networks, and volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Performance Considerations

### Container Resources
- **CPU**: Limit CPU usage if needed
- **Memory**: Set memory limits for production
- **Storage**: Use persistent volumes for model checkpoints

### Scaling Workers
To add more workers:
1. Update `docker-compose.yml` with new worker service
2. Update hub `--n-workers` parameter
3. Run `docker-compose up --build`

## Production Deployment

### Environment Variables
Create `.env` file:
```bash
N_ROUNDS=10
N_LOCAL_EPOCHS=20
N_WORKERS=4
PORT=8080
```

### Docker Compose with Environment Variables
```yaml
services:
  hub:
    environment:
      - N_ROUNDS=${N_ROUNDS:-5}
      - N_WORKERS=${N_WORKERS:-2}
```

### Health Checks
Add health checks to Docker Compose:
```yaml
services:
  hub:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```