import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter

import json
import re
from tqdm import tqdm
import random
import pickle
from fuzzywuzzy import fuzz

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util




### fast word-sentence similarity ###
# computer scores for fuzzy matching of word-sentence similarity
def fuzzy_score(sentence, word):
    return fuzz.partial_ratio(word.lower(), sentence.lower())

def fuzzy_scores(sentence, word_list):
    result = []
    for word in word_list:
        score = fuzz.partial_ratio(word.lower(), sentence.lower())
        result.append([word, score])
    return result    

# find top k words(i.e labels or names) yielding the highest fuzzy matching scores
def topk_lables(fuzzy_score_list, k = 5):
    fs_sort = sorted(fuzzy_score_list, key=lambda x: x[1], reverse=True)
    lbs = []
    if k < len(fs_sort):
        for i in range(k):
             lbs.append(fs_sort[i][0])
    else:
        for i in range(len(fs_sort)):
             lbs.append(fs_sort[i][0])
    return lbs
###-------------------------------### 





# def string_line_filter(string, filter_keywords, filter_out=True):
#     lines = string.splitlines()
#     filtered_lines = []
#     for line in lines:
#         append = True if filter_out else False
#         for keyword in filter_keywords:
#             if keyword in line:
#                 append = False if filter_out else True
#         if append:
#             filtered_lines.append(line)
#     new_string = "\n".join(filtered_lines)
    
#     return new_string

# def extract_table_keys(text):
#     lines = text.splitlines()
#     keys = []
#     for line in lines:
#         key = line.split(":", maxsplit=1)[0]        
#         keys.append(key.strip())
#     return keys

# def table_findall(text, keyword):
#     pattern = r'.*\b(' + "Camera" + r')\b.*'
#     matches = [line.strip() for line in text.split('\n') if re.match(pattern, line)]
#     return matches

### sentence-sentence similarity ###
def sentence_similarity(sentsim_model, text1, text2):
    embedding_1= sentsim_model.encode(text1, convert_to_tensor=True)
    embedding_2 = sentsim_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2)
###-------------------------------###


    
    
### Alpaca-HotPot (a.k.a Alpaca Fusion) model class
class AlpacaHotPotQA:
    def __init__(self, device, alpaca_model, tokenizer, phonedb_data, name_map):
        self.device = device
        self.prompt_template = ""
        self.prompter = Prompter(self.prompt_template)
        self.alp_model = alpaca_model
        self.tokenizer = tokenizer
        
        self.phonedb_data = phonedb_data
        self.name_map = name_map
        self.name_list = list(name_map.keys())
        
        # load BART model for zero-shot-classification
        self.bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # load sentence transformer model to get sentence embeddings and compute sentence-sentence similarity
        #  with a relatively fast inference speed compared to Alpaca itself
        self.sentsim_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    ### Perform phone related QA task using a fusion of the Alpaca LLM model and smaller classical NLP models
    def inference(self, question, print_process=True, yield_process = False):
        # check "name_query_mix_models_inference" function's definition written below for detailed documentation
        # this function returns all phone model name tokens found in the question sentence, it works even if
        # the sentence contain multiple phone names or the phone names is in a non-standard format
        key_names = self.name_query_mix_models_inference(question, self.name_list, print_process)

        if print_process:
            print("\n------------------------------------------------")
            print()
            print("Querying local DataBase ......")
        if yield_process:
            yield "Step 3: Query local database storing scraped text from phonedb.net", False
            yield "Querying local DataBase ......", False

        context_text = ""
        for n in key_names:
            # query the local phonedb text database
            keys_texts = self.query_key_text_list(n)
            relevant_texts = []

            if print_process:
                print("Model Name Family: ", n)
            if yield_process:
                yield "Model Name Family: "+str(n), False

            # too many texts, crashed the alpaca model, added a simple filter
            topk = 3
            keys_list = []
            texts_list = []
            for key, text in keys_texts:
                keys_list.append(key)
                texts_list.append(text)

            # use BART to do zero-shot multiple-choice classification, check the function def for details
            cls_res = self.efficient_bart_cls_inference(question, keys_list)
            keys_texts_ranked = []
            ranked_keys = cls_res["labels"][0 : topk]
            for i in range(len(keys_list)):
                if keys_list[i] in ranked_keys:
                    keys_texts_ranked.append([keys_list[i], texts_list[i]])


            for ln, text in keys_texts_ranked:
                relevant_text = "Model full name is '" + ln + "':\n"
                # use sentence-to-sentence cosine similarity scores to find relevant texts
                relevant_text += self.relevant_table_text(question, text, topk=3)
                relevant_text += "\n"
                #print(relevant_text)
                relevant_texts.append(relevant_text)
            sum_rele_text = ' '.join(relevant_texts)
            #print(sum_rele_text)


        if yield_process:
                yield "Step 4: Summarizing the text fetched using Alpaca base model", False

        if print_process:
            print("\n------------------------------------------------")
            print("Step 4: Summarizing the text fetched using Alpaca base model")
            
            # invoke Alpaca to summarize the long text
            instruction = '''Summarize the input passage which contains info about\
        different specific models of a cellphone family, don't omit model names
        '''
            input_text = sum_rele_text
            with torch.autocast("cuda"):
                output = self.alpaca_inference(input_text, instruction)

            output = output.split('###')[0].strip()
            output = output.strip()
            context_text += output + "\n\n" 
        

        if print_process:
            print("................................................")
            print("The summarized context information to be pipe into Alpaca model's prompt (Prompt Engineering)")
            print(context_text)

            print("------------------------------------------------")
        
        if yield_process:
            yield "The summarized context information to be pipe into Alpaca model's prompt:\n"+str(context_text), False

        # Final inference to answer the question
        instruction = "Answer the input question.\n\n\
        You are given the following context information extracted from local database:\n\n"
        instruction += context_text
        input_text = question
        with torch.autocast("cuda"):
            output = self.alpaca_inference(input_text, instruction)

        output = output.split('###')[0].strip()
        output = output.strip()

        if print_process:
            print("Step 5: Updating Alpaca's prompt with context info extracted from local database")
            print("        And finally instruct the model to answer the original user question")
            print("Back to the original question >>>>>> ")
            print("------------------------------------------------")
            print("Question:\n ", question)
            print("------------------------------------------------")
            print("\n\nAnswer:\n", output)
            print("------------------------------------------------")
        if yield_process:
            yield "Step 5: Updating Alpaca's prompt with context info extracted from local database:\n"+str(output), False
        
        if yield_process:
            yield output, True
            return output, True
        else:
            return output





    
    ### Call the Alpaca LLM model to perform inference, the inputs
    ### are primarily just an input text string and an instruction string,
    ### the instruction string will be wrapped by a internal prompter to
    ### couple with the input text to better instruct this LLM model
    ### for what to do with the input text, which is prompt-engineering for LLM
    ### ps: keep max_new_tokens value small if running on a GPU without a large GPU RAM   
    def alpaca_inference(self, input_text, instructions, 
        temperature = 0.3, top_p = 0.75, top_k = 40, num_beams = 1, 
        max_new_tokens = 666, **kwargs):

        input_prompt = self.prompter.generate_prompt(instructions, input_text)
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p,
            top_k=top_k, num_beams=num_beams, **kwargs)

        inputs = self.tokenizer(input_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            generation_output = self.alp_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return self.prompter.get_response(output)
    
    
    ### Use BART-based model to perform zero-shot multiple-choice classification,
    ### the input long_label_list are the choices label for the classification but
    ### exceeding the max allowed number of candidate labels, hence the topk_labels
    ### function filters the labels list by their fuzzy word-to-sentence scores  
    def efficient_bart_cls_inference(self, text, long_label_list):
        narrowed_labels = topk_lables(fuzzy_scores(text, long_label_list)) #narrowed down to short name list
        result = self.bart_classifier(text, narrowed_labels, multiclass=True)
        return result
    
    ### Helper function for querying local phonedb text database
    def query_specs_list(self, short_name, debug=False, replace_new_line = True):
        spec_list = []
        for ln in self.name_map[short_name]:
            if debug:
                print(ln)
            if replace_new_line:
                spec = self.phonedb_data[ln][0].replace("\\n", "\n")
            else:
                spec = self.phonedb_data[ln][0]
            spec_list.append(spec)
        return spec_list

    ### Same as above func but return a key-value pair instead,
    ### here the key is a short phone name such as "iPhone 13", the value
    ### is a list of all relevant text containing specs info for all relevant
    ### models such as "iPhone 13 mini, iPhone 13 Pro, iPhone 13 128GB, iPhone 13 256GB etc."
    def query_key_text_list(self, short_name, debug=False, replace_new_line = True):
        key_text = []
        for ln in self.name_map[short_name]:
            if debug:
                print(ln)
            if replace_new_line:
                spec = self.phonedb_data[ln][0].replace("\\n", "\n")
            else:
                spec = self.phonedb_data[ln][0]
            key_text.append([ln, spec])
        return key_text
    

    ### This function use Alpaca model to extract all phone model names appeared in a question, which
    ### are natural names that could take many different formats, such as iphone 13 pro, iPhone 13 pro 256G, 
    ### the pro version of iphone 13, Apple iphone 13 etc. which are all corresponding to the same name key 
    ### for querying local database. But in order to properly query local database, the output of Alpaca extraction 
    ### of the model name tokens must match exactly to the keys in the local name key list, which is difficult 
    ### for the alpaca model to complete everything as an end-to-end task due to a couple reasons: 
    ###     1. alpaca doesn't always generate consistent results with fixed format, 
    ###     2. the candidate key list is too long to be used to feed into the alpaca instruction prompt which 
    ###        has some limit on the the number of input tokens. 
    ###     3. One alpaca inference is time-consuming, which makes it very slow for alpaca inference to be
    ###        performed on each divided sub-tasks after dividing the long key list into shorter sublists
    ### As a result, we tried mixing alpaca and other smaller-and-less capable but faster NLP models. Specifically,
    ### the mix-models inference pipeline first use alpaca to extract arbitary number of model names appeared
    ### in the question sentence, and then select top 5 candidate name keys filtered from the local key list by
    ### ranking their the word-to-sentence fuzzy-matching similarity scores between candidate names and the question
    ### sentence. (experiments indicated fuzzy word-sentence scores are better than sentence-sentence embedding cos-similarity scores)   
    ### With the 5 candidate names, the task could be converted to an NLP multiple choice classification task,
    ### the zero-shot classification BART model is utilized to match every tokens extracted by earlier steps to
    ### the candidata names, and then the function returns those matched candidated names for accessing relevant
    ### text data in the local database
    ### 
    ### ps: BERT-based models performed poorly in zero-shot extracting multiple names appeared in a sentence,
    ### although fintuning them on a token-classification dataset on these phone model names should make 
    ### the BERT-based models to work properly, creating such dataset is time consuming and also defeating
    ### the generalization ability of the whole pipeline to extend to more general use case.
    def name_query_mix_models_inference(self, sentence, model_name_list, print_process = False):
        ### Step 1: Alpaca extract name tokens
        instruction = "Ignore the input. Extract all phone model names from the input sentence. \
    Append and prepend '%%%' symbols to each phone model name." 
        input_text = sentence
        if print_process:
            print("------------------------------------------------")
            print("Step 1: Alpaca extract name tokens")
            print("\n>>>>> Instruction:\n", instruction)
            print("\n>>>>> Input:\n", input_text)
            print("\nGenerating ......")

        with torch.autocast("cuda"):
            output = self.alpaca_inference(input_text, instruction, max_new_tokens = 128)

        output = output.split('###')[0].strip()
        output = output.strip()
        if print_process:
            print("\n<<<<< Output:\n", output)

        if print_process:
            print("\n------------------------------------------------")
            print("Using regex to tokenize:")
        matches = re.findall(r'%%([\w\s]+?)%%', output.replace("\n", ""))

        matches = list(set(matches))
        if print_process:
            print(matches)

        ### Step 2: iteratively call Bart classifier to get name keys for dict query
        ###         using the alpaca output as its input
        if print_process:
            print("------------------------------------------------")
            print("\nStep 2: Zero-shot BART classifier extract name keys\n")
        results = set()

        for token in matches:
            #Using fuzzy similarity scores to get top K candidate model names
            narrowed_labels = topk_lables(fuzzy_scores(token, model_name_list), k = 5) 

            cls_result = self.bart_classifier(token, narrowed_labels, multiclass=True)
            pred = cls_result["labels"][0] #get the label with the highest cls score
            results.add(pred)

            if print_process:
                print("Extracted Model Name: ", pred)

        if print_process:
            print("------------------------------------------------\n") 


        return results
    

    ### Use the classical word-embedding cosine similarity scores to find top K relevant context texts 
    ### within a text list
    def relevant_table_text(self, question, text, topk = 3):
        lines = text.splitlines()
        relevant_text = ""
        scores = []
        for line in tqdm(lines):
            score = sentence_similarity(self.sentsim_transformer, question, line).item()
            scores.append([line, score])
        scores.sort(key=lambda x: x[1], reverse=True)
        for i in range(topk):
            relevant_text += scores[i][0] + "\n"
        return relevant_text
    
    
    