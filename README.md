# Alpaca LLM based Cellphone Sales Chatbot

![](https://img.shields.io/badge/Linux%20build-pass-green.svg?logo=linux) 

![](https://img.shields.io/badge/NVIDIA-CUDA-green.svg?logo=nvidia) 

<img src="README.assets/image-20230510204636108-3722799.png" alt="image-20230510204636108" style="zoom:50%;" />

Domain-specific AI chatbot built with a combinations of Alpaca-lora and other classical NLP models such as BERT for cellphone sales relevant Q&amp;A.

Stanford Alpaca LLM(Large Language Model):https://crfm.stanford.edu/2023/03/13/alpaca.html

Alpaca-Lora model reference: https://github.com/tloen/alpaca-lora

Check [our presentation slides](https://github.com/zht043/cell-sales-chatbot/blob/main/ARIN7102%20presentation.pptx.pdf) for more details

Source Files:

* `chatbot.py` runs our finalized model with web UI 
* `alpaca_hotpot_qa.py` contain class definition of an alpaca & classical model fusion pipeline
* `alpaca-hotpot-qa-example-usages.ipynb` show usage example of using alpaca-hotpot

* `webscraping/` directory contains web-scraping code for crawling phonedb.net and techradar.com
* `model-weights/` stores our finetuned alpaca-lora for cellphone sales relevant Q&A tasks
* `bert/` [check this README](https://github.com/zht043/cell-sales-chatbot/blob/main/bert/README.md) 
* `bert2/` [check this README](https://github.com/zht043/cell-sales-chatbot/blob/main/bert2/readme.md)
* `small models` [check this README](https://github.com/zht043/cell-sales-chatbot/blob/main/small%20models/README.md)

## Install Dependencies

```
pip install -r requirements.txt
```

## Chatbot Web UI 

We utilize Gradio to construct a web user interface. Simply execute:

```shell
python chatbot.py
```

Optional parameters:

- `--concurrency_count <int>`: Set the concurrency count for the chatbot (default: 3).
- `--server_name <str>`: Specify the server name for Gradio (default: "0.0.0.0").
- `--server_port <int>`: Define the server port for Gradio. If not provided, Gradio will choose an available port starting from 7860.
- `--share`: Enable Gradio's sharing service (default: False).

![Web Interface](README.assets/web1.png)

## Alpaca-Hotpot Pipeline

Combining Alpaca-Lora LLM with other classical NLP models for QA task: Alpaca-Hotpot(Alpaca-Fusion)

<img src="README.assets/image-20230510202623722.png" alt="image-20230510202623722" style="zoom:50%;" /> 

### Walk-through of Alpaca-Hotpot pipelines

Question: **What is the maximum refresh rate of the iPhone 12 Pro Max display and how does it compare to the Samsung Galaxy Note20 Ultra?**

Answer with **step-by-step** print outs: 

```shell
------------------------------------------------
Step 1: Alpaca extract name tokens

>>>>> Instruction:
 Ignore the input. Extract all phone model names from the input sentence. Append and prepend '%%%' symbols to each phone model name.

>>>>> Input:
 What is the maximum refresh rate of the iPhone 12 Pro Max display and how does it compare to the Samsung Galaxy Note20 Ultra?

Generating ......

<<<<< Output:
 %%iPhone 12 Pro Max%% %%Samsung Galaxy Note20 Ultra%%

------------------------------------------------
Using regex to tokenize:
['iPhone 12 Pro Max', 'Samsung Galaxy Note20 Ultra']
------------------------------------------------

Step 2: Zero-shot BART classifier extract name keys

Extracted Model Name:  iPhone 12 Pro Max
Extracted Model Name:  Samsung Galaxy Note 20 Ultra
------------------------------------------------



------------------------------------------------
Querying local DataBase ......
Model Name Family:  iPhone 12 Pro Max
100%|██████████| 162/162 [00:01<00:00, 90.45it/s]
100%|██████████| 162/162 [00:01<00:00, 94.63it/s]
100%|██████████| 162/162 [00:01<00:00, 95.04it/s]
Model Name Family:  Samsung Galaxy Note 20 Ultra
100%|██████████| 167/167 [00:01<00:00, 89.10it/s]
100%|██████████| 167/167 [00:01<00:00, 94.14it/s]
------------------------------------------------
The iPhone 12 Pro Max 5G A2412 Dual SIM TD-LTE CN 512GB / A2413 has a 60Hz display refresh rate, a 120Hz touchscreen sampling rate, and a 60Hz display refresh rate. The iPhone 12 Pro Max 5G A2412 Dual SIM TD-LTE CN 256GB / A2413 has a 60Hz display refresh rate, a 120Hz touchscreen sampling rate, and a 60Hz display refresh rate. The iPhone 12 Pro Max 5G A2412 Dual SIM TD-LTE CN 128GB / A2413 has a 60Hz display refresh rate, a 120Hz touchscreen sampling rate, and a 60Hz display refresh rate.

The Samsung SM-N9860 Galaxy Note 20 Ultra 5G Dual SIM TD-LTE CN 512GB and the Samsung SM-N9860 Galaxy Note 20 Ultra 5G Dual SIM TD-LTE CN 256GB are two variants of the same phone, the Galaxy Note 20 Ultra 5G. Both variants have a 6.7-inch Super AMOLED display with a 120Hz refresh rate, a Snapdragon 865 chipset, and 512GB or 256GB of storage. The SM-N9860 Galaxy Note 20 Ultra 5G Dual SIM TD-LTE CN 512GB has a 512GB of storage, while the SM-N9860 Galaxy Note 20 Ultra 5G Dual SIM TD-LTE CN 256GB has a 256GB of storage. Both variants have a triple camera setup on the back, with a 48


------------------------------------------------
................................................
Back to the original question >>>>>> 
------------------------------------------------
Question:
  What is the maximum refresh rate of the iPhone 12 Pro Max display and how does it compare to the Samsung Galaxy Note20 Ultra?


Answer:
 The iPhone 12 Pro Max display has a maximum refresh rate of 60Hz. The Samsung Galaxy Note20 Ultra display has a maximum refresh rate of 120Hz.
```





Alpaca-Hotpot pipeline is defined as a class in [alpaca_hotpot_qa.py](https://github.com/zht043/cell-sales-chatbot/blob/main/alpaca_hotpot_qa.py), example usage can be found in this [alpaca-hotpot-qa-example-usages.ipynb](https://github.com/zht043/cell-sales-chatbot/blob/main/alpaca-hotpot-qa-example-usages.ipynb).

Example:

```python
from alpaca_hotpot_qa import AlpacaHotPotQA
alphot_qa_inference = AlpacaHotPotQA(device, alp_model, tokenizer, phonedb_data, name_map)

question = '''\
"How do the camera capabilities of the Apple iPhone 12, Samsung Galaxy S21, and Xiaomi Mi 11 compare?"
'''
answer = alphot_qa_inference.inference(question)
```



## Additional Demo Screenshots

![image-20230510204636108](README.assets/image-20230510204636108-3722799.png)

![image-20230510204553711](README.assets/image-20230510204553711.png)

![image-20230510204729312](README.assets/image-20230510204729312.png)
