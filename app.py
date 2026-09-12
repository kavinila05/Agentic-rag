import streamlit as st

from planner import check_document_scope
from rag import initialize_rag
from executor import run_agentic_rag


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Simple Agentic RAG",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title("🤖 Simple Agentic RAG")

st.write(
    "Upload a relevant document and ask a question. "
    "The agent will decide how to retrieve, evaluate, "
    "and use the information."
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a text document",
        type=["txt"]
    )

    document_ready = False


    # ------------------------------------------------
    # UPLOADED DOCUMENT
    # ------------------------------------------------

    if uploaded_file is not None:

        document_text = (
            uploaded_file
            .read()
            .decode("utf-8")
        )

        with st.spinner(
            "Checking document..."
        ):

            document_supported = (
                check_document_scope(
                    document_text
                )
            )


        if not document_supported:

            st.error(
                "❌ Unsupported document"
            )

            st.write(
                "This document does not appear to "
                "belong to the supported Tesla/business/"
                "financial domain."
            )

            st.caption(
                "Please upload a relevant document."
            )

        else:

            number_of_chunks = (
                initialize_rag(
                    document_text,
                    uploaded_file.name
                )
            )

            document_ready = True

            st.success(
                f"Loaded: {uploaded_file.name}"
            )

            st.caption(
                f"Created {number_of_chunks} chunks."
            )


    # ------------------------------------------------
    # DEFAULT DOCUMENT
    # ------------------------------------------------

    else:

        st.info(
            "No document uploaded. "
            "Using data/sample.txt."
        )

        document_ready = True


    st.divider()


    # ------------------------------------------------
    # CAPABILITIES
    # ------------------------------------------------

    st.header(
        "What the agent can do"
    )

    st.write(
        """
        • Check whether a question is supported

        • Check whether a document is relevant

        • Plan the question

        • Retrieve from documents

        • Evaluate retrieval

        • Rewrite failed queries

        • Use financial data

        • Use a calculator

        • Maintain shared state

        • Synthesize a grounded answer
        """
    )


# ==================================================
# QUESTION
# ==================================================

question = st.text_area(
    "Ask your question",

    placeholder=(
        "Example: What are the major risks "
        "that could affect Tesla's revenue?"
    ),

    height=120
)


ask_button = st.button(
    "🚀 Ask Agent",
    type="primary",
    use_container_width=True
)


# ==================================================
# EXECUTION
# ==================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    if not document_ready:

        st.warning(
            "Please upload a supported document."
        )

        st.stop()


    # ------------------------------------------------
    # RUN AGENT
    # ------------------------------------------------

    with st.spinner(
        "Agent is working..."
    ):

        result = run_agentic_rag(
            question
        )


    # ------------------------------------------------
    # OUT OF SCOPE
    # ------------------------------------------------

    if result["status"] == "out_of_scope":

        st.warning(
            "🚫 Question outside supported scope"
        )

        st.info(
            result["answer"]
        )

        st.stop()


    # ------------------------------------------------
    # PLANNING
    # ------------------------------------------------

    st.subheader(
        "🧠 Agent Planning"
    )

    plan = result["plan"]

    st.success(
        "Question accepted."
    )


    for i, subtask in enumerate(
        plan["subtasks"],
        1
    ):

        st.write(
            f"**{i}.** {subtask}"
        )


    # ------------------------------------------------
    # EXECUTION
    # ------------------------------------------------

    st.subheader(
        "⚙️ Agent Execution"
    )


    for i, subtask in enumerate(
        plan["subtasks"]
    ):

        with st.expander(
            f"Step {i + 1}: {subtask}"
        ):

            st.write(
                result["results"][i]
            )


    # ------------------------------------------------
    # FINAL ANSWER
    # ------------------------------------------------

    st.subheader(
        "📝 Final Answer"
    )

    st.markdown(
        result["answer"]
    )


    # ------------------------------------------------
    # AGENT STATE
    # ------------------------------------------------

    with st.expander(
        "🔍 View Agent State"
    ):

        for i, item in enumerate(
            result["results"],
            1
        ):

            st.markdown(
                f"### Result {i}"
            )

            st.write(
                item
            )