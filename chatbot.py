from load_models import *

import gradio as gr


import time
# def evaluate(*args, **kwargs):
#     rs = "### Instruction\n\ntest\n\n### Response:\n"
#     for i in "Hello!!!"+str(time.time())+"\n### Bad\n":
#         rs+=i
#         yield rs

import random

with gr.Blocks(title="Chatbot") as demo:
    gr.Markdown("## Online Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    
    def add_file(history, file):
        history = history + [((file.name,), None)]
        return history

    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def getInstruction(history):
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAs an AI model, you are reqired to generate response according to the chat history."
        # here add database log

        bert_qa_result = bert_qa_api("which kind of phones are suitable for gaming", "Some of the best phones for gaming include the Asus ROG Phone 5, Nubia Red Magic 6, iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, and OnePlus 9 Pro. Some of the best foldable phones available include the Samsung Galaxy Z Fold 2, Samsung Galaxy Z Flip, Huawei Mate X2, Xiaomi Mi Mix Fold, and Royole FlexPai 2. Some of the phones with the longest battery life include the Asus ZenFone 7, Samsung Galaxy M51, Moto G Power (2021), Xiaomi Poco X3 NFC, and Samsung Galaxy A72. Some of the best phones for people with hearing impairments include the iPhone 12 Pro Max, Samsung Galaxy S21 Ultra, Google Pixel 5, OnePlus 9 Pro, and Sony Xperia 1 II.")
        prompt+="Here is some hints: "+bert_qa_result



        if len(history)>1:
            prompt+="\nThe history chat is:\n"
            for i in history[:-1]:
                prompt+="\nInput:\n"+i[0]+"\nResponse:\n"+i[1]+"\n" if type(i[0]) == str else "\nInput:\n"+i[1]+"\n"

        prompt+="\n\n### Input:\n"+history[-1][0]+"\n\n### Response:\n" 
        print(prompt)
        return prompt

    def callchat(history, log, temperature, top_p, top_k, num_beams, max_new_tokens):
        log = time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "start generating prompt\n"; yield {log_box: log}
        history[-1][1] = ""

        prompt = getInstruction(history)
        log += time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "start streaming\n"; yield {log_box: log}
        for current_response, decodeded_output in alpaca_evaluate(prompt,temperature, top_p, top_k, num_beams, max_new_tokens):

            history[-1][1] = current_response
            yield {chatbot:history, developer_box:decodeded_output}
        log += time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "finished\n"
        yield {log_box: log}

    def bot(history):
        bot_message = random.choice(["User uploaded a picture.", "I am still learning to recognize pictures.", "I'm still learning."])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            yield history
        
    with gr.Row():
        with gr.Column(scale=0.85):
            submit_btn = gr.Button(value="Submit",variant="primary")
        with gr.Column(scale=0.15, min_width=0):
            upload_btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
        with gr.Column(scale=0.15, min_width=0):
            clear = gr.Button("Clear")
    
    with gr.Accordion("Open for More!", open=False):
        with gr.Row():
            temperature = gr.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            )
            top_k = gr.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            )
            num_beams = gr.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams"
            )
            max_new_tokens = gr.Slider(
                minimum=16, maximum=2000, step=1, value=256, label="Max tokens"
            )
        with gr.Row():
            developer_box = gr.TextArea(label="Original Output",interactive = True,lines = 10)
            log_box = gr.TextArea(label="Log Output",interactive = True,lines = 10)

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        callchat, [chatbot, log_box, temperature, top_p, top_k, num_beams, max_new_tokens], [chatbot, developer_box, log_box]
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        callchat, [chatbot, log_box, temperature, top_p, top_k, num_beams, max_new_tokens], [chatbot, developer_box, log_box]
    )
    upload_btn.upload(add_file, [chatbot, upload_btn], [chatbot]).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue(concurrency_count=3)
if __name__ == "__main__":
    demo.launch(server_name = server_name, server_port = server_port, share_gradio = share_gradio)

