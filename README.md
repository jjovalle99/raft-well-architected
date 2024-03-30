# Going In Depth Into: RAFT: Adapting Language Model to Domain Specific RAG

The authors of this method describe RAFT as:

*RAFT is a training strategy designed to enhance the model's performance in answering questions within a specific domain, in "open-book" settings. This technique demonstrates a fine-tuning recipe for LLMs for question-answering tasks based on a selected collection of documents. We have pinpointed several crucial design decisions, such as training the model alongside distractor documents, organizing the dataset so a portion lacks oracle documents in their context, and formulating answers in a chain-of-thought manner with direct quotations from the relevant text*

You can take a look at the authors material [here](https://gorilla.cs.berkeley.edu/blogs/9_raft.html).

In this project I decided to test their methodology in step by step fashion. Despite they have already some scripts automated for this, I am always encouragin to understand the detalis in order to fully take advantage of this awesome apporaches and innovations.

The execution of the in-depth execution of RAFT was made with the following steps:
- 