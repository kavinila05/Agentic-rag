from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


# ==================================================
# LLM
# ==================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ==================================================
# SIMPLE PLAN PARSER
# ==================================================

def parse_plan(text: str):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    subtasks = []

    needs_documents = False
    needs_finance = False
    needs_calculator = False

    for line in lines:

        lower = line.lower()

        if lower.startswith("documents:"):

            needs_documents = (
                "yes" in lower
            )

        elif lower.startswith("finance:"):

            needs_finance = (
                "yes" in lower
            )

        elif lower.startswith("calculator:"):

            needs_calculator = (
                "yes" in lower
            )

        elif (
            line[0:2].isdigit()
            and "." in line[:4]
        ):

            task = line.split(
                ".", 1
            )[1].strip()

            if task:
                subtasks.append(task)

    return {
        "subtasks": subtasks,
        "needs_documents": needs_documents,
        "needs_finance": needs_finance,
        "needs_calculator": needs_calculator
    }


# ==================================================
# QUESTION SCOPE CHECK
# ==================================================

def check_question_scope(
    question: str
):

    prompt = f"""
You are the scope checker for a small Agentic RAG
proof-of-concept.

SUPPORTED QUESTIONS:

- Questions about the uploaded document
- Tesla/business information
- Business risks
- Revenue
- Financial information
- Financial comparisons
- Calculations involving the above

The system is NOT a general-purpose assistant.

OUT OF SCOPE:

- Medical questions
- Symptoms
- Diagnosis
- Mental health
- Personal advice
- Relationships
- Weather
- Sports
- Unrelated news
- Coding questions
- Random general knowledge
- Casual conversation
- "How are you?"
- "I am not feeling well"

Return ONLY:

YES

if the question belongs to the supported domain.

Return ONLY:

NO

if it does not.

USER QUESTION:

{question}
"""

    response = llm.invoke(
        prompt
    )

    answer = (
        response.content
        .strip()
        .upper()
    )

    return answer.startswith("YES")


# ==================================================
# DOCUMENT SCOPE CHECK
# ==================================================

def check_document_scope(
    document_text: str
):

    # Only send a reasonable amount to the LLM.
    sample = document_text[:12000]

    prompt = f"""
You are checking whether a document is suitable
for a small Agentic RAG proof-of-concept.

SUPPORTED DOCUMENT CONTENT:

- Tesla
- Electric vehicles
- Automotive businesses
- Tesla business risks
- Revenue
- Financial information
- Business analysis
- Information directly related to these topics

A document does NOT need to mention Tesla specifically
if it is clearly an automotive/business/financial document
that can reasonably support questions in this system.

UNSUPPORTED EXAMPLES:

- requirements.txt
- software dependency lists
- unrelated programming documentation
- recipes
- novels
- medical documents
- personal documents
- random unrelated text

Return ONLY:

YES

if the document is relevant to the supported domain.

Return ONLY:

NO

if it is unrelated.

DOCUMENT:

{sample}
"""

    response = llm.invoke(
        prompt
    )

    answer = (
        response.content
        .strip()
        .upper()
    )

    return answer.startswith("YES")


# ==================================================
# PLANNER
# ==================================================

def plan_question(
    question: str
):

    prompt = f"""
You are the planning component of a SIMPLE AGENTIC RAG
system.

The question has already passed the scope check.

Create the smallest useful execution plan.

AVAILABLE INFORMATION SOURCES:

1. Documents
   Information from the currently loaded document.

2. Finance
   Mock financial data available through the finance tool.

3. Calculator
   Arithmetic and numerical calculations.

IMPORTANT:

Only create subtasks that are actually necessary.

Return EXACTLY this format:

Documents: YES or NO
Finance: YES or NO
Calculator: YES or NO

1. first subtask
2. second subtask
3. third subtask

Do not add explanations.

USER QUESTION:

{question}
"""

    response = llm.invoke(
        prompt
    )

    return parse_plan(
        response.content
    )