from typing import List, TypedDict

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, START, END

from src.prompt import (
    system_prompt,
    doc_grader_prompt,
    query_rewrite_prompt,
    hallucination_grader_prompt,
    answer_grader_prompt,
)


class GraphState(TypedDict):
    """
    Represents the state of the CRAG graph.

    question: user question
    documents: relevant (possibly corrected) context documents
    generation: LLM generated answer
    rewritten_query: rewritten question used for web search fallback
    max_retries: max correction retries allowed
    retry_count: correction retries used so far
    """

    question: str
    documents: List[Document]
    generation: str
    rewritten_query: str
    max_retries: int
    retry_count: int


class GradeBinaryAnswer(BaseModel):
    """Binary relevance score of a retrieved document against the question."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class GradeHallucination(BaseModel):
    """Binary score whether the generation is grounded in the documents."""

    binary_score: str = Field(description="Grounded score 'yes' or 'no'")


class GradeAnswerUsefulness(BaseModel):
    """Binary score whether the answer addresses the question."""

    binary_score: str = Field(description="Usefulness score 'yes' or 'no'")


def build_crag_chain(retriever, llm):
    """
    Build and compile the Corrective RAG (CRAG) graph.

    retrieve -> grade_documents -> (transform_documents | rewrite_query -> web_search)
             -> generate -> grade_generation -> (END | rewrite_query)
    """

    # Graders with structured output
    doc_grader_llm = llm.with_structured_output(GradeBinaryAnswer)
    hallucination_grader_llm = llm.with_structured_output(GradeHallucination)
    answer_grader_llm = llm.with_structured_output(GradeAnswerUsefulness)

    # Prompts
    doc_grader_prompt_tpl = ChatPromptTemplate.from_messages(
        [("human", doc_grader_prompt)]
    )
    query_rewrite_prompt_tpl = ChatPromptTemplate.from_messages(
        [("human", query_rewrite_prompt)]
    )
    hallucination_prompt_tpl = ChatPromptTemplate.from_messages(
        [("human", hallucination_grader_prompt)]
    )
    answer_prompt_tpl = ChatPromptTemplate.from_messages(
        [("human", answer_grader_prompt)]
    )
    generate_prompt_tpl = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    web_search_tool = DuckDuckGoSearchResults(num_results=3)

    # ------------------------- Nodes -------------------------

    def retrieve(state: GraphState):
        """Retrieve the top-k documents for the question."""
        question = state["question"]
        print(f"[CRAG] retrieve: {question}")
        documents = retriever.invoke(question)
        return {"documents": documents}

    def grade_documents(state: GraphState):
        """Filter out retrieved documents not relevant to the question."""
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for doc in documents:
            grade = doc_grader_llm.invoke(
                doc_grader_prompt_tpl.invoke(
                    {"question": question, "document": doc.page_content}
                )
            )
            if grade.binary_score.lower() == "yes":
                filtered_docs.append(doc)
                print(f"[CRAG] grade_documents: kept doc from {doc.metadata.get('source')}")
            else:
                print(f"[CRAG] grade_documents: dropped doc from {doc.metadata.get('source')}")

        return {"documents": filtered_docs}

    def decide_to_generate(state: GraphState):
        """Route to generation when relevant docs exist, else correct via web search."""
        if state["documents"]:
            print("[CRAG] decide_to_generate: generate from retrieved docs")
            return "transform_documents"
        print("[CRAG] decide_to_generate: no relevant docs -> rewrite + web search")
        return "rewrite_query"

    def transform_documents(state: GraphState):
        """Light compaction of kept chunks for a cleaner generation prompt."""
        compacted = []
        for doc in state["documents"]:
            content = " ".join(doc.page_content.split())
            compacted.append(Document(page_content=content, metadata=doc.metadata))
        return {"documents": compacted}

    def rewrite_query(state: GraphState):
        """Rewrite the question to improve search results."""
        question = state["question"]
        print(f"[CRAG] rewrite_query: {question}")
        rewritten = llm.invoke(
            query_rewrite_prompt_tpl.invoke({"question": question})
        ).content
        return {"rewritten_query": rewritten.strip()}

    def web_search(state: GraphState):
        """DuckDuckGo fallback; search results become the generation context."""
        query = state["rewritten_query"]
        print(f"[CRAG] web_search: {query}")
        results = web_search_tool.invoke(query)
        web_doc = Document(page_content=str(results), metadata={"source": "web"})
        return {"documents": [web_doc], "retry_count": state.get("retry_count", 0) + 1}

    def generate(state: GraphState):
        """Generate the answer from the (possibly corrected) context."""
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(doc.page_content for doc in documents)
        generation = llm.invoke(
            generate_prompt_tpl.invoke({"context": context, "input": question})
        ).content
        return {"generation": generation}

    # ------------------------- Conditional edges -------------------------

    def decide_after_grading(state: GraphState):
        """If the generation failed checks and retries remain, correct it."""
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        context = "\n\n".join(doc.page_content for doc in documents)

        hallucination = hallucination_grader_llm.invoke(
            hallucination_prompt_tpl.invoke(
                {"documents": context, "generation": generation}
            )
        )
        if hallucination.binary_score.lower() == "no":
            if retry_count < max_retries:
                print("[CRAG] decide_after_grading: hallucinated -> rewrite query")
                return "rewrite_query"
            print("[CRAG] decide_after_grading: hallucinated but retries exhausted")
            return END

        usefulness = answer_grader_llm.invoke(
            answer_prompt_tpl.invoke(
                {"question": question, "generation": generation}
            )
        )
        if usefulness.binary_score.lower() == "no" and retry_count < max_retries:
            print("[CRAG] decide_after_grading: not useful -> rewrite query")
            return "rewrite_query"

        print("[CRAG] decide_after_grading: answer accepted")
        return END


    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_documents", transform_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        ["transform_documents", "rewrite_query"],
    )
    workflow.add_edge("transform_documents", "generate")
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        decide_after_grading,
        ["rewrite_query", END],
    )

    return workflow.compile()
