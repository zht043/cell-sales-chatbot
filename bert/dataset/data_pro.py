import json
with open("/home/yiming/Documents/HKU-DASC7606-A2-main/dataset/train.json", 'r') as f:
    data = json.load(f)

title = data['data'][0]['title']
paragraphs = data['data'][0]['paragraphs']

n = 0
for (index_i,i) in enumerate(paragraphs):
    context = i['context']
    qas = i['qas']
    for (index_j,j) in enumerate(qas):
        question = j['question']
        id = j['id']
        answers = j['answers']
        text = answers[0]['text']
        print(data['data'][0]['paragraphs'][index_i]['qas'][index_j]['answers'][0]['answer_start'])
        data['data'][0]['paragraphs'][index_i]['qas'][index_j]['answers'][0]['answer_start'] = context.find(text)
        data['data'][0]['paragraphs'][index_i]['qas'][index_j]['id'] = str(index_i) + 'X' + str(index_j) + 'N' + str(id)

with open("/home/yiming/Documents/HKU-DASC7606-A2-main/dataset/train.json", 'w') as f:
    json.dump(data,f, indent=4)