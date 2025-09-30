# TTS Model Integration Guide

## XTTS-v2 Integration

```python
# In xtts_loader.py, replace mock with:
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def load_model(self, model_path: str = None):
    config = XttsConfig()
    config.load_json(model_path or "path/to/xtts/config.json")
    self.model = Xtts.init_from_config(config)
    self.model.load_checkpoint(config, checkpoint_path="path/to/checkpoint.pth")
```

## OpenVoice Integration

```python
# In openvoice_loader.py, replace mock with:
import openvoice

def load_model(self, model_path: str = None):
    self.model = openvoice.load_model(model_path or "checkpoints/base_speakers/EN")
```

## Model Downloads

- XTTS-v2: `huggingface-cli download coqui/XTTS-v2`
- OpenVoice: `git clone https://github.com/myshell-ai/OpenVoice`