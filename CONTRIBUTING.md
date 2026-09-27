# Contributing

Thank you for contributing to GenAI Text-to-Image Studio.

## Development setup

1. Fork or clone the repository.
2. Create and activate a virtual environment.
3. Install a compatible PyTorch build for your machine.
4. Install application dependencies:

```bash
python -m pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run app.py
```

## Before opening a pull request

Please:

- Keep changes focused and small where practical.
- Preserve the existing modular architecture.
- Avoid hard-coded personal paths.
- Never commit model weights, caches, secrets, or generated local outputs.
- Update documentation when behavior or setup changes.
- Run the repository checks before submitting.

## Pull requests

Describe:

- What changed
- Why it changed
- How it was tested
- Any hardware or deployment considerations

## Issues

For bugs, include the Python version, relevant package versions, operating system, hardware information when relevant, and a reproducible description.

Please do not include API keys, access tokens, private data, or other secrets in issues.
