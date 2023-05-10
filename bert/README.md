# Closed Domain And Open Domain Q&A with BERT
[![logo](https://img.shields.io/badge/HUANGYming-projects-orange?style=flat&logo=github)](https://github.com/HUANGYming) 

![](https://img.shields.io/badge/Linux%20build-pass-green.svg?logo=linux) 
![](https://img.shields.io/badge/NVIDIA-CUDA-green.svg?logo=nvidia) 

## Table of Contents

- [Structure](#Structure)
- [Installation](#installation)
- [Closed Domain Q&A Mode](#closed-domain)
- [Open Domain Q&A Mode](#open-domain)
- [Dataset](#dataset)
- [License](#license)

## I. Structure <a id="Structure"></a>

```
unet-multiclass-pytorch/
    - dataset/
    - output/
    - README.md
    - bert_cdqa_api.py
    - bert_cdqa_api_without_retrieval.py
    - inference.py
    - inference.sh
    - inference_without_retrieval.py
    - inference_without_retrieval.sh
    - match_document.py
    - modeling_bert.py
    - utils_squad.py
    - utils_squad_evaluate.py
    - requirements.txt
```

in which:

- `dataset/` store the dataset

- `output/` contains the model and inference result

- `README.md` contains the guidance

- `bert_cdqa_api.py` contains the entrance of API of closed-domain Q&A

- `bert_cdqa_api_without_retrieval.py` contains the entrance of API of open-domain Q&A

- `inference_without_retrieval.py` contains the inference function

- `inference_without_retrieval.sh` contains the inference entrance through shell

- `match_document.py` make retrieval by TF-IDF

- `modeling_bert.py` contains the basic structure of BERT 

- `utils_squad.py` contains the necessary functions of BERT

- `utils_squad_evaluate.py` contains the necessary functions of BERT

- `requirements.txt` contains the necessary packages

  

## II. Installation <a id="installation"></a>
Trained Model [Download link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/huangym2_connect_hku_hk/EYmpO2yt_VdEsqVN5lVBhgcB3r0jPWiU7U-Lkn-GdVFESA?e=FOQ5wT) (Put the model into output/)
```
matplotlib==3.2.2
numpy==1.24.3
pandas==2.0.0
pytorch==2.0.0
torchvision==0.15.0
torchaudio==2.0.0
pytorch-cuda==11.7
tensorboard==2.6.0
tqdm==4.65.0
```

To install for Ubuntu,
```
$ conda install -r requirements.txt
```

## III. Closed Domain QA Mode <a id="closed-domain"></a>

In this mode, Context will be automatically searched from document database by Retriever based on TF-IDF. So, user only need to input Question.

### 1. Preview in shell

```shell
sh inference.sh Question 
```

For example:

```shell
sh inference.sh "which kind of phones are suitable for gaming" 
```

where:

```
Question = "which kind of phones are suitable for gaming"
```

Output:

```
Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro
```

### 2. API in Python

```python
from inference import main
question = "which kind of phones are suitable for gaming" 
result = main(question)
```

Output:

```
Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro
```
## IV. Open Domain QA Mode <a id="open-domain"></a>

In this mode, Context that including the answer of Question need to be given manually. So, user  need to input Question and Context.

### By Shell

To run by shell:

```shell
sh inference_without_retrieval.sh Question Context
```

For example:

```shell
sh inference_without_retrieval.sh "which kind of phones are suitable for gaming" "Some of the best phones for gaming include the Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro. Some of the best foldable phones available include the Samsung Galaxy Z Fold 2, Samsung Galaxy Z Flip, Huawei Mate X2, Xiaomi Mi Mix Fold, and Royole FlexPai 2. Some of the phones with the longest battery life include the Asus ZenFone 7, Samsung Galaxy M51, Moto G Power (2021), Xiaomi Poco X3 NFC, and Samsung Galaxy A72. Some of the best phones for people with hearing impairments include the iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, OnePlus 9 Pro, and Sony Xperia 1 II."
```

where :

```
Question = "which kind of phones are suitable for gaming"
Context = "Some of the best phones for gaming include the Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro. Some of the best foldable phones available include the Samsung Galaxy Z Fold 2, Samsung Galaxy Z Flip, Huawei Mate X2, Xiaomi Mi Mix Fold, and Royole FlexPai 2. Some of the phones with the longest battery life include the Asus ZenFone 7, Samsung Galaxy M51, Moto G Power (2021), Xiaomi Poco X3 NFC, and Samsung Galaxy A72. Some of the best phones for people with hearing impairments include the iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, OnePlus 9 Pro, and Sony Xperia 1 II."
```

Output:

```
Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro
```

### API in Python

````python
from inference_without_retrieval import main
question = "which kind of phones are suitable for gaming"
context = "Some of the best phones for gaming include the Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro. Some of the best foldable phones available include the Samsung Galaxy Z Fold 2, Samsung Galaxy Z Flip, Huawei Mate X2, Xiaomi Mi Mix Fold, and Royole FlexPai 2. Some of the phones with the longest battery life include the Asus ZenFone 7, Samsung Galaxy M51, Moto G Power (2021), Xiaomi Poco X3 NFC, and Samsung Galaxy A72. Some of the best phones for people with hearing impairments include the iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, OnePlus 9 Pro, and Sony Xperia 1 II."
result = main(question, context)
````

Output:

```python
Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro
```

## Dataset <a id="dataset"></a>

Based on phone1.json from Ning Zichun

#### phone1.json:

```json
{
    "instruction": "What are the best budget phones under $200?",
    "input": "",
    "output": "Some of the best budget phones under $200 include the Moto G Play (2021), Nokia 2.4, Samsung Galaxy A01, LG K40, and Xiaomi Redmi 9A."
}
```

In preprocessing, the format converted to the same as SQuAD 2.0. Also, the questions and answers are generated by chatGPT-4 based on the context of phone1.json. The "answer_start" is recalculated due to the error of chatGPT-4.

```json
{
     "context": "Some of the best phones for photography include the iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, Huawei P40 Pro, and OnePlus 9 Pro.",
     "qas": [
         {
             "question": "Which phones are considered among the best for photography?",
             "id": "1X0N1",
             "answers": [
                 {
                     "text": "iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, Huawei P40 Pro, and OnePlus 9 Pro",
                     "answer_start": 52
                 }
             ]
         },
     ]
}
```

## License <a id="License "></a>

MIT © HYM

