# Step 1. Run FastChat server
python3 -m fastchat.serve.controller --host 0.0.0.0

# Step 2. Run model worker
# Replace --model-path with the path to the model checkpoint
python3 -m fastchat.serve.model_worker --model-names "retroformer" --model-path lmsys/longchat-13b-16k --host 0.0.0.0 --port 31001 --worker http://localhost:31001

# Step 3. Run OpenAI proxy/
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000
