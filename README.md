1) venv + `pip install -f requirements`
2) install ollama and `ollama run llama3` to download it.
3) Set a system prompt in `ConversationManager::__init__`
4) Change cpu and int8 to cuda and float16 if you have a nice gpu
