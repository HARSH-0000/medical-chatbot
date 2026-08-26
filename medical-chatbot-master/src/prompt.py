
system_prompt = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five to ten sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


# CRAG: grade relevance of a retrieved document to the question
doc_grader_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

Here is the retrieved document: 

 {document} 


Here is the user question: {question}"""


# CRAG: rewrite the question for better web search results
query_rewrite_prompt = """Look at the input and try to reason about the underlying semantic intent / meaning.
Here is the initial question:

 ------- 
 {question} 
 ------- 

Formulate an improved question:"""


# CRAG: check whether the generation is grounded in the retrieved context
hallucination_grader_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

Here are the retrieved facts: 

 {documents} 


Here is the LLM generation: {generation}"""


# CRAG: check whether the answer actually addresses the question
answer_grader_prompt = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.

Here is the user question: {question} \n
Here is the LLM generation: {generation}"""
