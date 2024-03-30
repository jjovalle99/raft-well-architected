# Going In Depth Into: RAFT: Adapting Language Model to Domain Specific RAG

The authors of this method describe RAFT as:

_RAFT is a training strategy designed to enhance the model's performance in answering questions within a specific domain, in "open-book" settings. This technique demonstrates a fine-tuning recipe for LLMs for question-answering tasks based on a selected collection of documents. We have pinpointed several crucial design decisions, such as training the model alongside distractor documents, organizing the dataset so a portion lacks oracle documents in their context, and formulating answers in a chain-of-thought manner with direct quotations from the relevant text._

You can take a look at the authors' material [here](https://gorilla.cs.berkeley.edu/blogs/9_raft.html).

In this project, I decided to test their methodology in a step-by-step fashion. Despite the fact that they already have some scripts automated for this, I am always encouraged to understand the details in order to fully take advantage of these awesome approaches and innovations.

The execution of this manual RAFT was made with the following steps:

| Stage                       | Notebook/Script                | Tech Stack                |
|-----------------------------|--------------------------------|--------------------------------|
| Creating RAFT Style Dataset | [RAFT-Dataset-StepbyStep.ipynb](https://github.com/jjovalle99/raft-well-architected/blob/866df6af5f87c011594c2b271066703767f34040/RAFT-Dataset-StepbyStep.ipynb)  | Llama-Parse, LangChain, HuggingFace Datasets |
| RAFT fine-tuning            | [RAFT-Finetuning-Starling7b](https://github.com/jjovalle99/raft-well-architected/blob/866df6af5f87c011594c2b271066703767f34040/RAFT-Finetuning-Starling7b.ipynb)     | Transformers, PEFT, TRL | 
| RAFT model deployment       | [src/serve_model.py](https://github.com/jjovalle99/raft-well-architected/blob/866df6af5f87c011594c2b271066703767f34040/model/serve_model.py)             | vLLM, Modal | 
| RAFT model evaluation       | [RAFT-Model-Comparison.ipynb](https://github.com/jjovalle99/raft-well-architected/blob/866df6af5f87c011594c2b271066703767f34040/RAFT-Model-Comparison.ipynb)    | LangSmith|