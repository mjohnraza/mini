# 1. Run full automated test suite (8/8 tests pass)
python -m pytest -v

# 2. Evaluate model parameters, layers, and validation perplexity
python -m experiments.evaluate

# 3. Text Generation - Shakespearean Prompt
python -m src.generate --prompt "ROMEO:" --max-new-tokens 70 --temperature 0.8

# 4. Text Generation - Modern / AI Prompt
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80 --temperature 0.8

# 5. Launch FastAPI live microservice
# Open browser at http://127.0.0.1:8000/docs
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
