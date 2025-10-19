# EFCA-ADAPT-AG: Advanced AGI Meta-Reinforcement Learning Agent

Enables rapid adaptation to new tasks by learning an initialization that can be quickly fine-tuned with few gradient steps.

## Table of Contents
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Configuration

### Configuration File Location

The application uses YAML configuration files located in the `configs/` directory:
- **Default Configuration**: `configs/default.yaml`
- **Custom Configurations**: You can create additional configuration files for different environments (e.g., `configs/production.yaml`, `configs/development.yaml`)

### Configuration File Example

Here's a complete example of the configuration file structure:

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 4
  log_level: "info"

# Model Configuration
model:
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 2048
  top_p: 0.9
  frequency_penalty: 0.0
  presence_penalty: 0.0
  timeout: 60

# API Configuration
api:
  rate_limit: 100  # requests per minute
  max_retries: 3
  retry_delay: 1.0  # seconds
  enable_cors: true
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8080"

# Processing Configuration
processing:
  batch_size: 32
  max_queue_size: 1000
  timeout: 300  # seconds
  enable_async: true
  num_workers: 4

# Feature Flags
features:
  enable_caching: true
  enable_metrics: true
  enable_tracing: false
  enable_debugging: false
```

### Configuration Parameters Explained

#### Server Settings
- `host`: Server bind address (default: "0.0.0.0" for all interfaces)
- `port`: Port number for the API server (default: 8000)
- `reload`: Enable auto-reload on code changes (development only)
- `workers`: Number of worker processes for handling requests
- `log_level`: Logging verbosity (debug, info, warning, error, critical)

#### Model Settings
- `name`: Model identifier to use for inference
- `temperature`: Controls randomness in generation (0.0 to 1.0)
- `max_tokens`: Maximum number of tokens in the response
- `timeout`: Request timeout in seconds

#### API Settings
- `rate_limit`: Maximum requests per minute per client
- `enable_cors`: Enable Cross-Origin Resource Sharing
- `cors_origins`: List of allowed origins for CORS

### Loading Configuration

To specify a configuration file when running the application:

```bash
# Use default configuration
python ai_studio_code.py --config configs/default.yaml

# Use custom configuration
python ai_studio_code.py --config configs/production.yaml
```

## API Documentation

### FastAPI Endpoints

The application provides a RESTful API built with FastAPI. All endpoints are documented with OpenAPI/Swagger.

#### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if the service is running and healthy.

**Sample Request**:
```bash
curl -X GET http://localhost:8000/health
```

**Sample Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-19T16:18:00Z",
  "version": "1.0.0"
}
```

---

#### 2. Generate Completion

**Endpoint**: `POST /api/v1/generate`

**Description**: Generate text completion based on a prompt.

**Sample Request**:
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Explain meta-learning in simple terms",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Request Body**:
```json
{
  "prompt": "string (required)",
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": ["\n\n"],
  "stream": false
}
```

**Sample Response**:
```json
{
  "id": "gen-abc123",
  "object": "text_completion",
  "created": 1697724000,
  "model": "gpt-4",
  "choices": [
    {
      "text": "Meta-learning is a technique where AI models learn how to learn. Instead of training a model for one specific task, meta-learning enables the model to quickly adapt to new tasks with minimal additional training.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 45,
    "total_tokens": 53
  }
}
```

---

#### 3. Batch Processing

**Endpoint**: `POST /api/v1/batch`

**Description**: Process multiple requests in a single batch.

**Sample Request**:
```bash
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "requests": [
      {"prompt": "What is MAML?", "max_tokens": 100},
      {"prompt": "Explain PPO algorithm", "max_tokens": 100}
    ]
  }'
```

**Sample Response**:
```json
{
  "batch_id": "batch-xyz789",
  "results": [
    {
      "id": "gen-001",
      "text": "MAML (Model-Agnostic Meta-Learning) is a meta-learning algorithm...",
      "status": "completed"
    },
    {
      "id": "gen-002",
      "text": "PPO (Proximal Policy Optimization) is a reinforcement learning algorithm...",
      "status": "completed"
    }
  ],
  "total_processed": 2,
  "total_time_ms": 1250
}
```

---

#### 4. Model Training

**Endpoint**: `POST /api/v1/train`

**Description**: Trigger a meta-learning training session.

**Sample Request**:
```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "dataset": "custom_tasks",
    "epochs": 100,
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "num_inner_steps": 5
  }'
```

**Sample Response**:
```json
{
  "training_id": "train-abc123",
  "status": "started",
  "estimated_time_minutes": 45,
  "config": {
    "dataset": "custom_tasks",
    "epochs": 100,
    "inner_lr": 0.01,
    "outer_lr": 0.001
  }
}
```

---

#### 5. Get Metrics

**Endpoint**: `GET /api/v1/metrics`

**Description**: Retrieve current system metrics and statistics.

**Sample Request**:
```bash
curl -X GET http://localhost:8000/api/v1/metrics \
  -H "X-API-Key: your-api-key"
```

**Sample Response**:
```json
{
  "requests": {
    "total": 15420,
    "success": 15180,
    "failed": 240,
    "rate_per_minute": 45.2
  },
  "latency": {
    "mean_ms": 234.5,
    "p50_ms": 198.2,
    "p95_ms": 456.7,
    "p99_ms": 782.1
  },
  "model": {
    "active_models": 1,
    "cache_hit_rate": 0.78,
    "avg_tokens_per_request": 127.3
  },
  "system": {
    "cpu_usage_percent": 42.5,
    "memory_usage_mb": 2048.7,
    "uptime_hours": 72.3
  }
}
```

---

### Interactive API Documentation

Once the server is running, you can access interactive API documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to:
- Explore all available endpoints
- View detailed request/response schemas
- Test API calls directly from your browser
- Download OpenAPI specification

### Authentication

The API supports API key authentication. Include your API key in the request header:

```bash
-H "X-API-Key: your-api-key-here"
```

To enable authentication, set `security.enable_auth: true` in your configuration file.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for accelerated training

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/efca-adapt-ag.git
cd efca-adapt-ag

# Install required packages
pip install -r requirements.txt
```

## Usage

### Running the Server

```bash
# Start with default configuration
python ai_studio_code.py --mode server

# Start with custom configuration
python ai_studio_code.py --mode server --config configs/production.yaml

# Start with custom port
python ai_studio_code.py --mode server --port 8080
```

### Training Mode

```bash
# Run training
python ai_studio_code.py --mode train --config configs/default.yaml
```

### Evaluation Mode

```bash
# Run evaluation
python ai_studio_code.py --mode eval --config configs/default.yaml
```

### PPO (Proximal Policy Optimization)

Provides stable policy updates through clipped objective functions, balancing exploration and exploitation.

### Curiosity-Driven Learning

Augments extrinsic rewards with intrinsic motivation based on prediction error, encouraging exploration of novel states.

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests and documentation. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{efca_adapt_ag,
  author = {Sunghun Kwag},
  title = {EFCA-ADAPT-AG: Advanced AGI Meta-Reinforcement Learning Agent},
  year = {2025},
  url = {https://github.com/sunghunkwag/efca-adapt-ag}
}
```

## Contact

**Project Maintainer**: Sunghun Kwag
- GitHub: [@sunghunkwag](https://github.com/sunghunkwag)
- Repository: [efca-adapt-ag](https://github.com/sunghunkwag/efca-adapt-ag)

For questions, issues, or collaboration opportunities, please open an issue on GitHub or reach out directly.

## Acknowledgments

This project builds upon foundational work in meta-learning and reinforcement learning. Special thanks to the research community for their contributions to MAML, PPO, and curiosity-driven learning algorithms.

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: October 2025
