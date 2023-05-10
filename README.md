# Alpaca LLM based Cellphone Sales Chatbot
Domain-specific AI chatbot built with a combinations of Alpaca-lora and other classical NLP models such as BERT for cellphone sales relevant Q&amp;A.

Stanford Alpaca model:

Implementation reference: https://github.com/tloen/alpaca-lora



Install Dependencies:

```
pip install -r requirements.txt
```



## Chatbot Web Interface

We utilize Gradio to construct a web interface. Simply execute:

```shell
python chatbot.py
```

Optional parameters:

- `--concurrency_count <int>`: Set the concurrency count for the chatbot (default: 3).
- `--server_name <str>`: Specify the server name for Gradio (default: "0.0.0.0").
- `--server_port <int>`: Define the server port for Gradio. If not provided, Gradio will choose an available port starting from 7860.
- `--share`: Enable Gradio's sharing service (default: False).

![Web Interface](./img/web1.png)

## Alpaca-Hotpot Pipeline

Combining Alpaca-Lora 
