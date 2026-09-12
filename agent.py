from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from rag import search_documents
from finance import finance_lookup
from planner import plan_question


load_dotenv()


# --------------------------------------------------
# 1. LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# 2. Calculator
# --------------------------------------------------

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Use this tool when arithmetic or numerical
    calculations are required.
    """

    try:

        allowed = set("0123456789+-*/(). ")

        if not all(char in allowed for char in expression):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Could not calculate the expression."


# --------------------------------------------------
# 3. Create the agent
# --------------------------------------------------

tools = [
    search_documents,
    finance_lookup,
    calculator
]


agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are the execution agent in an Agentic RAG system.

You receive a user question together with an execution plan.

Your job is to EXECUTE the plan.

Available tools:

1. search_documents

Search the local Tesla document collection.

Use this for information contained in the
local documents.

2. finance_lookup

Retrieve structured financial data.

Use this for financial numbers such as:

- revenue
- net income
- company financial data
- quarter-specific financial data

3. calculator

Perform mathematical calculations.

IMPORTANT:

Follow the execution plan.

If the plan requires finance data:

    Use finance_lookup.

If the plan requires document information:

    Use search_documents.

If the plan requires a calculation:

    Use calculator.

You may call the same tool multiple times.

For example, if the plan requires:

- Tesla Q2 2026 revenue
- Industry average Q2 2026 revenue

then call finance_lookup separately for both.

Do not skip a required subtask.

Do not invent information.

The finance tool contains MOCK DATA FOR THIS POC.
Do not describe those numbers as real Tesla financial
filings.

For document information, use information actually
returned by search_documents.

After completing all subtasks, combine the results
into one clear final answer.

If information is unavailable, say so clearly.
"""
)


# --------------------------------------------------
# 4. Run the planned agent
# --------------------------------------------------

question = """
Compare Tesla's Q2 2026 revenue with the industry
average, and explain which risks in our documents
could affect Tesla's revenue.
"""


# --------------------------------------------------
# 5. Create the plan first
# --------------------------------------------------

print("\n============================")
print("PLANNING")
print("============================")

plan = plan_question(question)

for i, task in enumerate(plan.subtasks, start=1):
    print(f"{i}. {task}")

print("\nRequired sources:")
print(f"Documents : {plan.needs_documents}")
print(f"Finance   : {plan.needs_finance}")
print(f"Calculator: {plan.needs_calculator}")


# --------------------------------------------------
# 6. Convert the plan into agent instructions
# --------------------------------------------------

plan_text = "\n".join(
    f"{i}. {task}"
    for i, task in enumerate(plan.subtasks, start=1)
)


execution_message = f"""
User question:

{question}


Execution plan:

{plan_text}


Required sources:

Documents required: {plan.needs_documents}
Finance required: {plan.needs_finance}
Calculator required: {plan.needs_calculator}


Execute this plan completely.

Use the appropriate tools for each subtask.

Do not skip subtasks.
"""


# --------------------------------------------------
# 7. Execute the plan
# --------------------------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": execution_message
            }
        ]
    }
)


# --------------------------------------------------
# 8. Display the agent's process
# --------------------------------------------------

print("\n============================")
print("AGENT EXECUTION")
print("============================")

for message in result["messages"]:

    print(f"\n{message.type.upper()}:")

    if message.content:
        print(message.content)

    tool_calls = getattr(message, "tool_calls", [])

    if tool_calls:
        print("Tool calls:")
        print(tool_calls)


# --------------------------------------------------
# 9. Final answer
# --------------------------------------------------

print("\n============================")
print("FINAL ANSWER")
print("============================")

print(result["messages"][-1].content)