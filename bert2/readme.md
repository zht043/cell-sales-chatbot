dataset
======
dataset consists of three parts: instructions, inputs, outputs;
use outputs as answer text

usage
=====

    import bert_QA.pickle

    with open("bert_QA.pickle","rb")as f:
        func = pickle.load(f)
    func(dataset_path,question)
