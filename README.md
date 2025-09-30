# Catalyst AI Voice Studio

A professional-grade open-source Python monorepo for voice synthesis, comparable in quality to ElevenLabs and NotebookLM.

## Features

- **Modular TTS Backends**: Support for XTTS-v2, OpenVoice, Fish-Speech (future), StyleTTS-2 (research)
- **Extensible Architecture**: Prosody planning, text normalization, and real-time streaming
- **Production Ready**: FastAPI web service with streaming support
- **Fine-tuning Support**: Tools for dataset preparation and model fine-tuning
- **Comprehensive Testing**: Unit tests, benchmarks, and evaluation metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/catalyst-ai/voice-studio.git
cd voice-studio

# Install dependencies
pip install -r requirements.txt

# Or install with optional GPU support
pip install -e ".[gpu]"
```

### Basic Usage

```python
from catalyst_ai_voice_studio.tts_service import XTTSLoader

# Load TTS model
tts = XTTSLoader()
tts.load_model()

# Synthesize speech
audio = tts.synthesize("Hello, world!", voice_id="default")
```

### Web API

```bash
# Start the FastAPI server
uvicorn catalyst_ai_voice_studio.web_streamer.api:app --reload

# Test synthesis endpoint
curl -X POST "http://localhost:8000/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!", "voice_id": "default"}'
```

## Architecture

```
catalyst_ai_voice_studio/
├── tts_service/          # Core TTS engine loaders
├── prosody_planner/      # Prosody and emphasis planning
├── text_normalizer/      # Text preprocessing and phonemization
├── web_streamer/         # FastAPI web service
├── tts_finetune/         # Fine-tuning tools and scripts
└── utils/                # Shared utilities
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Start development server
make run
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.