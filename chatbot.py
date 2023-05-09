from load_models import *

if __name__=="__main__":    # args parser

    import gradio as gr
    import sys
    import time
    import random

    # Set default values
    concurrency_count = 3
    server_name = "0.0.0.0"
    server_port = None
    share = False


    with gr.Blocks(title="Chatbot") as demo:    # Gradio Web Interface

        def user(user_message, history):    # Immediate funtion to add new line to chat history
            return "", history + [[user_message, "Waiting for prompt to be generated..."]]

        def callchat(history, log, temperature, top_p, top_k, num_beams, max_new_tokens):   # Chatbot main function, stream response

            #   START of prompt generation
            log = time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "start generating prompt: bert\n"; yield {log_box: log}
            history[-1][1] = ""

            ##  Part 1: Begin of the prompt
            prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAs an AI model, you are reqired to generate response according to the chat history."
            
            ##  Part 2: Call BERT to get related information
            bert_qa_result = bert_qa_api(history[-1][0],from_api = True)
            if bert_qa_result:
                prompt+="\nHere are some hints: "+bert_qa_result
                log += str(bert_qa_result)+"\n"; yield {log_box: log}
            
            ##  Part 3: Call ALPHOT to get related information
            log += time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "start generating prompt: alphot\n"; yield {log_box: log}

            for new_log, status_code in alphot_qa_inference.inference(history[-1][0],yield_process=True):
                if not status_code:
                    log += str(new_log)+"\n"
                    yield {log_box: log}
                else:
                    alphot_qa_result = new_log
            if alphot_qa_result:
                prompt+="\nHere are some hints: "+alphot_qa_result

            ##  Part 4: Add history chat log to the prompt
            if len(history)>1:
                prompt+="\nThe history chat is:\n"
                for i in history[:-1]:
                    prompt+="\nInput:\n"+i[0]+"\nResponse:\n"+i[1]+"\n" if type(i[0]) == str else "\nInput:\n"+i[1]+"\n"

            prompt+="\n\n### Input:\n"+history[-1][0]+"\n\n### Response:\n" 
            print(prompt)
            #   END of prompt generation

            #   START of response streaming 
            log += time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "start streaming\n"; yield {log_box: log}

            for current_response, decodeded_output in alpaca_evaluate(prompt,temperature, top_p, top_k, num_beams, max_new_tokens):
                history[-1][1] = current_response
                yield {chatbot:history, developer_box:decodeded_output}
            
            log += time.strftime("%d %b %Y %H:%M:%S: ", time.gmtime(time.time())) + "finished\n"
            yield {log_box: log}
            #   END of response streaming 

        def add_file(history, file):    # dummy function (not implemented)
            history = history + [((file.name,), None)]
            return history
        
        def bot(history):   # dummy function
            bot_message = random.choice(["User uploaded a picture.", "I am still learning to recognize pictures.", "I'm still learning."])
            history[-1][1] = ""
            for character in bot_message:
                history[-1][1] += character
                yield history


        #   START of interface
        gr.Markdown("## Online Chatbot")
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
            
        with gr.Row():
            with gr.Column(scale=0.85): # call chatbot
                submit_btn = gr.Button(value="Submit",variant="primary")
            with gr.Column(scale=0.15, min_width=0):    # dummy
                upload_btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
            with gr.Column(scale=0.15, min_width=0):    # clear history
                clear = gr.Button("Clear")
        
        with gr.Accordion("Open for More!", open=True): # more parameters, raw output and log
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
                developer_box = gr.TextArea(label="Original Output",interactive = True,lines = 10)  # raw output (not stripped)
                log_box = gr.TextArea(label="Log Output",interactive = True,lines = 10) # pipeline log

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then( # textbox submit callback (when user press Enter/Return key)
            callchat, [chatbot, log_box, temperature, top_p, top_k, num_beams, max_new_tokens], [chatbot, developer_box, log_box]
        )
        submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(   # chatbot submit button callback
            callchat, [chatbot, log_box, temperature, top_p, top_k, num_beams, max_new_tokens], [chatbot, developer_box, log_box]
        )
        upload_btn.upload(add_file, [chatbot, upload_btn], [chatbot]).then( # dummy button callback
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)   # clear button callback
        


    if "--concurrency_count" in sys.argv:
        concurrency_count = int(sys.argv[sys.argv.index("--concurrency_count") + 1])

    if "--server_name" in sys.argv:
        server_name = sys.argv[sys.argv.index("--server_name") + 1]

    if "--server_port" in sys.argv:
        server_port = int(sys.argv[sys.argv.index("--server_port") + 1])

    if "--share" in sys.argv:
        share = True

demo.queue(concurrency_count = concurrency_count)
demo.launch(server_name = server_name, server_port = server_port, share = share)

