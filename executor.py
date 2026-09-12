import time

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from rag import search_documents
from finance import finance_lookup
from planner import plan_question


load_dotenv()


# ==================================================
# LLM
# ==================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ==================================================
# SAFE LLM CALL
# ==================================================

def call_llm(
    prompt: str,
    retries: int = 2
):

    for attempt in range(
        retries + 1
    ):

        try:

            return llm.invoke(
                prompt
            )

        except Exception as e:

            if "429" not in str(e):
                raise

            if attempt == retries:
                raise

            print(
                "Groq rate limit reached. "
                "Waiting 3 seconds..."
            )

            time.sleep(3)


# ==================================================
# CALCULATOR
# ==================================================

@tool
def calculator(
    expression: str
) -> str:
    """
    Calculate a mathematical expression.
    """

    try:

        allowed = set(
            "0123456789+-*/(). "
        )

        if not all(
            char in allowed
            for char in expression
        ):

            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Could not calculate the expression."


# ==================================================
# RETRIEVAL EVALUATION
# ==================================================

def evaluate_retrieval(
    question: str,
    retrieved_information: str
):

    if (
        "RETRIEVAL_STATUS: NO_RESULTS"
        in retrieved_information
    ):

        return False, (
            "No useful information was retrieved."
        )

    if (
        "RETRIEVAL_STATUS: NO_DOCUMENT"
        in retrieved_information
    ):

        return False, (
            "No document is loaded."
        )

    prompt = f"""
You are evaluating retrieval for an Agentic RAG system.

QUESTION:

{question}

RETRIEVED INFORMATION:

{retrieved_information}

Does the retrieved information contain enough
useful evidence to answer the question?

Return ONLY:

YES

or

NO
"""

    response = call_llm(
        prompt
    )

    answer = (
        response.content
        .strip()
        .upper()
    )

    if answer.startswith("YES"):

        return True, (
            "Retrieved information is sufficient."
        )

    return False, (
        "Retrieved information is insufficient."
    )


# ==================================================
# QUERY REWRITING
# ==================================================

def rewrite_query(
    question: str,
    previous_query: str,
    previous_result: str
):

    prompt = f"""
Rewrite the search query for a document retrieval system.

Original question:

{question}

Previous query:

{previous_query}

Previous result:

{previous_result}

The previous retrieval was insufficient.

Create ONE more specific search query.

Return ONLY the query.
"""

    response = call_llm(
        prompt
    )

    query = (
        response.content
        .strip()
    )

    return query or question


# ==================================================
# AGENT
# ==================================================

tools = [
    search_documents,
    finance_lookup,
    calculator
]


agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are an execution agent inside a small Agentic RAG system.

TOOLS:

search_documents
- Search the currently loaded document.

finance_lookup
- Retrieve mock financial data.

calculator
- Perform arithmetic.

RULES:

Use search_documents for document questions.

Use finance_lookup for financial numbers.

Use calculator for calculations.

Use previous subtask results when useful.

Do not invent facts.

Do not use outside information.

The finance tool contains MOCK DATA FOR THIS POC.
Never present those numbers as real financial filings.
"""
)


# ==================================================
# DOCUMENT SUBTASK DETECTION
# ==================================================

def is_document_subtask(
    subtask: str
):

    terms = [
        "document",
        "documents",
        "local",
        "risk",
        "risks",
        "retrieve",
        "identify",
        "extract",
        "evidence"
    ]

    text = subtask.lower()

    return any(
        term in text
        for term in terms
    )


# ==================================================
# DOCUMENT EXECUTION
# ==================================================

def execute_document_subtask(
    subtask: str,
    previous_results: list[str],
    max_retries: int = 3
):

    query = subtask

    for attempt in range(
        max_retries
    ):

        retrieval = (
            search_documents.invoke(
                {
                    "query": query
                }
            )
        )

        relevant, reason = (
            evaluate_retrieval(
                subtask,
                retrieval
            )
        )

        print(
            f"\nRetrieval attempt {attempt + 1}"
        )

        print(
            f"Relevant: {relevant}"
        )

        print(
            f"Reason: {reason}"
        )

        if relevant:

            return retrieval

        if attempt == max_retries - 1:

            return (
                "RETRIEVAL_FAILED_AFTER_RETRIES\n\n"
                f"{retrieval}"
            )

        query = rewrite_query(
            subtask,
            query,
            retrieval
        )

        print(
            f"Rewritten query: {query}"
        )

    return (
        "No useful retrieval result found."
    )


# ==================================================
# NORMAL EXECUTION
# ==================================================

def execute_normal_subtask(
    subtask: str,
    previous_results: list[str]
):

    if previous_results:

        previous_context = "\n\n".join(
            f"RESULT {i + 1}:\n{result}"
            for i, result in enumerate(
                previous_results
            )
        )

    else:

        previous_context = (
            "No previous results."
        )

    prompt = f"""
CURRENT SUBTASK:

{subtask}

PREVIOUS RESULTS:

{previous_context}

Complete the current subtask.

Use the appropriate tools.

Do not invent information.
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    return response[
        "messages"
    ][-1].content


# ==================================================
# EXECUTE SUBTASK
# ==================================================

def execute_subtask(
    subtask: str,
    previous_results: list[str]
):

    if is_document_subtask(
        subtask
    ):

        return execute_document_subtask(
            subtask,
            previous_results
        )

    return execute_normal_subtask(
        subtask,
        previous_results
    )


# ==================================================
# FINAL SYNTHESIS
# ==================================================

def synthesize_answer(
    question: str,
    results: list[str]
):

    evidence = "\n\n".join(
        f"RESULT {i + 1}:\n{result}"
        for i, result in enumerate(
            results
        )
    )

    prompt = f"""
You are the final answer component of an Agentic RAG system.

USER QUESTION:

{question}

VERIFIED RESULTS:

{evidence}

Answer using ONLY the verified results.

Rules:

- Do not invent facts.
- Do not use outside information.
- Document claims must come from retrieved evidence.
- Finance numbers are MOCK DATA FOR THIS POC.
- Clearly identify mock financial data.
- If the evidence does not answer the question,
  say that the information is not available.
"""

    response = call_llm(
        prompt
    )

    return response.content


# ==================================================
# COMPLETE WORKFLOW
# ==================================================

def run_agentic_rag(
    question: str
):

    # ------------------------------------------------
    # SCOPE CHECK
    # ------------------------------------------------

    from planner import check_question_scope

    in_scope = check_question_scope(
        question
    )

    if not in_scope:

        return {
            "status": "out_of_scope",

            "reason": (
                "This question is outside the "
                "supported domain."
            ),

            "answer": (
                "I can't help with that question. "
                "This Agentic RAG is limited to "
                "document-based Tesla/business and "
                "financial questions."
            ),

            "plan": None,

            "results": []
        }


    # ------------------------------------------------
    # PLAN
    # ------------------------------------------------

    plan = plan_question(
        question
    )


    # ------------------------------------------------
    # SHARED STATE
    # ------------------------------------------------

    state = {
        "subtask_results": []
    }


    # ------------------------------------------------
    # EXECUTE
    # ------------------------------------------------

    for subtask in plan["subtasks"]:

        result = execute_subtask(
            subtask,
            state["subtask_results"]
        )

        state[
            "subtask_results"
        ].append(
            result
        )


    # ------------------------------------------------
    # SYNTHESIS
    # ------------------------------------------------

    final_answer = synthesize_answer(
        question,
        state["subtask_results"]
    )


    return {
        "status": "success",

        "reason": "Question accepted.",

        "answer": final_answer,

        "plan": plan,

        "results": state[
            "subtask_results"
        ]
    }