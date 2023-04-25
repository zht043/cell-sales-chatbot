import os
import sys

import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


import traceback
from queue import Queue
from threading import Thread



class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


load_8bit= True
base_model = 'decapoda-research/llama-7b-hf'
lora_weights: str = 'model-weights/alpaca-phone'
# Allows to listen on all interfaces by providing '0.
server_name: str = "0.0.0.0",
share_gradio: bool = False
server_port: None

base_model = base_model or os.environ.get("BASE_MODEL", "")
assert (
    base_model
), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
    )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)



def evaluate(   # alpaca evaluate function
    prompt,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=256,
    stream_output=True,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break
                if decoded_output.endswith("\n### "):   # early stop
                    yield decoded_output[:-4]
                    return
                else:
                    yield decoded_output
        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield decoded_output.split("### Response:" )[1].strip()

# def evaluate(*args, **kwargs):
#     rs = "### Instruction\n\ntest\n\n### Response:\n"
#     for i in "Hello!!!"+str(time.time())+"\n### Bad\n":
#         rs+=i
#         yield rs

import gradio as gr
import random
import time

with gr.Blocks() as demo:
    gr.Markdown("## Online Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    
    def add_file(history, file):
        history = history + [((file.name,), None)]
        return history

    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def getInstruction(history):
        prompt = "### Instruction:\nYou are an AI assistant that happy to solve any question.\
Below is an instruction paired with an input that provides further context. \
Write a response that appropriately completes the request."
        # here add database log

        if len(history)>1:
            prompt+="\nThe history chat is:\n"
            for i in history[:-1]:
                prompt+="\nInput:\n"+i[0]+"\nResponse:\n"+i[1]+"\n" if type(i[0]) == str else "\nInput:\n"+i[1]+"\n"

        prompt+="\n\n### Input:\n"+history[-1][0]+"\n\n### Response:\n" 
        print(prompt)
        return prompt

    def callchat(history,temperature, top_p, top_k, num_beams, max_new_tokens):
        history[-1][1] = ""
        for i in evaluate(getInstruction(history),temperature, top_p, top_k, num_beams, max_new_tokens):

            current_response = i.split("### Response:" )[1].strip()

            history[-1][1] = current_response
            yield history, i

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
                minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
            )
        with gr.Row():
            developer_box = gr.TextArea(label="Original Output")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        callchat, [chatbot, temperature, top_p, top_k, num_beams, max_new_tokens], [chatbot,developer_box]
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        callchat, [chatbot, temperature, top_p, top_k, num_beams, max_new_tokens], [chatbot,developer_box]
    )
    upload_btn.upload(add_file, [chatbot, upload_btn], [chatbot]).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue(concurrency_count=3)
if __name__ == "__main__":
    demo.launch(server_name = server_name, server_port = server_port, share_gradio = share_gradio)

