# Run in Closed Domain QA Mode

 In this mode, Context will be automatically searched from document database by Retriever based on TF-IDF. So, user only need to input Question.

## Preview in shell

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



## API in Python

```python
from inference import main
question = "which kind of phones are suitable for gaming" 
result = main(question)
```

Output:

```
Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro
```



# Run in Open Domain QA Mode 

 In this mode, Context that including the answer of Question need to be given manually. So, user  need to input Question and Context.

## By Shell

To run by shell:

```shell
sh inference.sh Question Context
```

For example:

```shell
sh inference.sh "which kind of phones are suitable for gaming" "Some of the best phones for gaming include the Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro. Some of the best foldable phones available include the Samsung Galaxy Z Fold 2, Samsung Galaxy Z Flip, Huawei Mate X2, Xiaomi Mi Mix Fold, and Royole FlexPai 2. Some of the phones with the longest battery life include the Asus ZenFone 7, Samsung Galaxy M51, Moto G Power (2021), Xiaomi Poco X3 NFC, and Samsung Galaxy A72. Some of the best phones for people with hearing impairments include the iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, OnePlus 9 Pro, and Sony Xperia 1 II."
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



## API in Python

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
