import streamlit as st
import os
import json
import re
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import io
from docx import Document

# =========== For LLM Integration ===========
from langchain_openai import ChatOpenAI

# openai_api = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")
openai_api = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(temperature=0, api_key=openai_api, model="gpt-4o")

# =========== For Embeddings & Vector Store ===========
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    collection_name="company_rfp_collection",
    embedding_function=embeddings,
    persist_directory='chroma_db_dir'
)

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

# =========== LangGraph and Workflow Logic ===========
from langgraph.graph import Graph
from langchain.prompts import PromptTemplate
from typing import Dict, Any

StateType = Dict[str, Any]

def extract_or_generate_questions(state: StateType) -> StateType:
    rfp_doc = state["rfp_doc"]
    prompt_template = PromptTemplate.from_template(
        """
        You are an RFP Expert. We have an RFP document below.
        If it contains explicit questions (section-wise), extract them faithfully.
        If no questions are found, generate a list of 8-12 relevant questions 
        that a client would typically expect in an RFP for these requirements.
        Make them fairly detailed.

        RFP Document:
        {rfp_doc}

        Output format (JSON only, no extra text):
        {{
          "questions": [
            {{
              "section": "Section Heading or Number",
              "question_text": "The actual question found or generated"
            }},
            ...
          ]
        }}
        """
    )
    prompt = prompt_template.format(rfp_doc=rfp_doc)
    response = llm.invoke(prompt).content

    cleaned_response = re.sub(r"```(?:json)?\n?", "", response).strip().replace("```", "")
    try:
        parsed_json = json.loads(cleaned_response)
    except json.JSONDecodeError:
        parsed_json = {"questions": []}

    questions = parsed_json.get("questions", [])
    if not questions:
        raise ValueError("No questions could be extracted or generated from the RFP document.")

    state["questions"] = questions
    return state

def refine_answer(question: str, original_answer: str, user_feedback: str) -> str:
    """
    Calls the LLM with a prompt containing the question + original answer + feedback,
    returns a refined answer string.
    """
    refine_prompt_template = PromptTemplate.from_template(
        """
        You are refining an RFP answer. The user provided feedback, and you have the question and original answer.

        Question: {question}
        Original Answer: {original_answer}
        Feedback: {feedback}

        Please revise the answer text to address the feedback, 
        preserving any crucial company details and providing a thorough, relevant explanation.

        Respond in JSON only, with the format:
        {{
          "answer_text": "Revised final answer"
        }}
        """
    )
    prompt_text = refine_prompt_template.format(
        question=question,
        original_answer=original_answer,
        feedback=user_feedback
    )

    # Suppose we have a "llm" object for the LLM:
    llm_response = llm.invoke(prompt_text).content

    cleaned_response = re.sub(r"```(?:json)?\n?", "", llm_response).strip().replace("```", "")
    try:
        parsed_json = json.loads(cleaned_response)
        return parsed_json.get("answer_text", llm_response)
    except json.JSONDecodeError:
        return llm_response
    
def answer_questions_iteratively(state: StateType) -> StateType:
    questions = state.get("questions", [])
    if not questions:
        raise ValueError("No questions found in state['questions']. Please run 'extract_or_generate_questions' first.")

    final_answers = []
    for q in questions:
        query = q["question_text"]
        docs = vectorstore.similarity_search(query, k=2)
        context_text = "\n\n".join([d.page_content for d in docs])

        prompt_template = PromptTemplate.from_template(
            """
            You are a specialized RFP writer for our company.
            We have the following previously submitted proposals (context):
            {company_context}

            Now, craft a *very detailed* (at least several paragraphs) response 
            to the RFP question below, ensuring it reflects our company's 
            capabilities, solutions, prior experience, and references.
            Tailor the language to sound professional and thorough.

            Question:
            {question}

            Format (JSON only):
            {{
              "answer_text": "Your long-form answer here."
            }}
            """
        )
        prompt = prompt_template.format(
            company_context=context_text,
            question=q["question_text"]
        )

        llm_response = llm.invoke(prompt).content
        cleaned_llm_response = re.sub(r"```(?:json)?\n?", "", llm_response).strip().replace("```", "")
        try:
            parsed_answer = json.loads(cleaned_llm_response)
            answer_text = parsed_answer.get("answer_text", llm_response)
        except json.JSONDecodeError:
            answer_text = llm_response

        final_answers.append({
            "section": q["section"],
            "question_text": q["question_text"],
            "answer_text": answer_text
        })

    state["answers"] = final_answers
    return state

def incorporate_user_feedback(state: StateType) -> StateType:
    feedback_dict = state.get("feedback_dict", {})
    answers = state.get("answers", [])

    for idx, feedback in feedback_dict.items():
        if idx < len(answers):
            original_answer = answers[idx]["answer_text"]
            refine_prompt_template = PromptTemplate.from_template(
                """
                The user provided feedback to refine the answer. 
                Original Answer:
                {original_answer}

                Feedback:
                {feedback}

                Please revise the answer text to address the feedback, 
                preserving any crucial company details. 
                Make it even more thorough if possible.

                Format (JSON only):
                {{
                  "answer_text": "Revised final answer"
                }}
                """
            )
            refine_prompt = refine_prompt_template.format(
                original_answer=original_answer,
                feedback=feedback
            )
            llm_response = llm.invoke(refine_prompt).content
            cleaned_llm_response = re.sub(r"```(?:json)?\n?", "", llm_response).strip().replace("```", "")
            try:
                parsed_refined = json.loads(cleaned_llm_response)
                refined_answer = parsed_refined.get("answer_text", llm_response)
            except json.JSONDecodeError:
                refined_answer = llm_response
            
            answers[idx]["answer_text"] = refined_answer

    state["answers"] = answers
    return state

def compile_rfp_response(state: StateType) -> StateType:
    final_answers = state.get("answers", [])
    compiled_sections = []
    for qa in final_answers:
        section_heading = f"## {qa['section']}\n\n" if qa.get("section") else ""
        q_text = f"**Q**: {qa['question_text']}\n"
        a_text = f"**A**: {qa['answer_text']}\n"
        compiled_sections.append(section_heading + q_text + a_text + "\n")

    final_doc = "\n".join(compiled_sections)
    state["final_rfp_response"] = final_doc
    return state

def create_word_doc_from_text(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

def compile_rfp_document(answers):
    """
    Builds a string-based final doc from the list of Q&As.
    Then we can convert that to .docx for download.
    """
    compiled = []
    for idx, qa in enumerate(answers):
        section_line = f"## {qa.get('section','')}\n\n" if qa.get('section') else ""
        question_line = f"**Q{idx+1}**: {qa['question_text']}\n"
        answer_line = f"**A**: {qa['answer_text']}\n"
        compiled.append(section_line + question_line + answer_line)
    return "\n".join(compiled)

# Build the Graph
workflow = Graph()
workflow.add_node("extract_or_generate_questions", extract_or_generate_questions)
workflow.add_node("answer_questions_iteratively", answer_questions_iteratively)
workflow.add_node("incorporate_user_feedback", incorporate_user_feedback)
workflow.add_node("compile_rfp_response", compile_rfp_response)

workflow.add_edge("extract_or_generate_questions", "answer_questions_iteratively")
workflow.add_edge("answer_questions_iteratively", "incorporate_user_feedback")
workflow.add_edge("incorporate_user_feedback", "compile_rfp_response")

workflow.set_entry_point("extract_or_generate_questions")
g=workflow
app = g.compile()

# ====================== STREAMLIT UI BELOW ======================
def page_index_existing_responses():
    st.header("Index Existing Company RFP Responses")
    uploaded_files = st.file_uploader(
        "Upload your company's past RFP responses (PDF or TXT)", 
        accept_multiple_files=True, type=["txt","pdf"]
    )
    if uploaded_files:
        for file in uploaded_files:
            file_bytes = file.read()
            if file.type == "text/plain":
                content = file_bytes.decode("utf-8", errors="ignore")
            else:
                content = f"PDF parsing not implemented in this demo for file {file.name}."

            chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
            metadata_list = [{"filename": file.name} for _ in chunks]
            vectorstore.add_texts(chunks, metadata=metadata_list)
        st.success("Successfully indexed the uploaded documents into Chroma vector DB!")

def page_run_workflow():
    st.header("RFP Workflow")

    # Step A: user provides RFP doc
    rfp_text_input = st.text_area("Paste your RFP template content here", height=300)
    uploaded_rfp = st.file_uploader("Or upload a PDF/TXT with the RFP content", type=["txt","pdf"])

    if uploaded_rfp is not None:
        if uploaded_rfp.type == "text/plain":
            rfp_text_input = uploaded_rfp.read().decode("utf-8", errors="ignore")
        else:
            rfp_text_input = f"PDF parsing not implemented in this demo for file {uploaded_rfp.name}."

    if st.button("Process RFP"):
        if not rfp_text_input.strip():
            st.error("Please provide RFP content (pasted or uploaded).")
            return

        # Initialize state
        st.session_state["state"] = {}
        st.session_state["state"]["rfp_doc"] = rfp_text_input
        st.session_state["state"]["feedback_dict"] = {}

        try:
            # 1) Extract or generate questions
            new_state = extract_or_generate_questions(st.session_state["state"])
            # 2) Answer them with RAG
            new_state = answer_questions_iteratively(new_state)

            # Save results
            st.session_state["state"] = new_state
            st.session_state["all_questions"] = new_state["questions"]
            st.session_state["answers"] = new_state["answers"]
            st.session_state["finalized"] = [False]*len(st.session_state["answers"])
            st.session_state["num_finalized"] = 0

            st.success("Questions extracted and initial answers generated. Scroll down to review them.")
        except Exception as e:
            st.error(f"Error while processing RFP: {e}")

    # Display Q&A if available
    if "answers" not in st.session_state:
        st.warning("Please run 'Process RFP' first to populate Q&A.")
        return

    answers = st.session_state["answers"]

    st.write("### Review & Refine Answers")

    for idx, qa in enumerate(answers):
        st.write("---")
        st.markdown(f"**Question {idx+1}**")
        st.markdown(f"Section: {qa.get('section','N/A')}")
        st.markdown(f"Question: {qa['question_text']}")
        st.markdown(f"Answer: {qa['answer_text']}")

        # We'll use 'feedback_{idx}_clear' as a boolean to see if we should clear
        fb_key = f"feedback_{idx}"
        clear_flag_key = f"{fb_key}_clear"

        # If the clear flag is set, reset the text area state first
        if st.session_state.get(clear_flag_key, False):
            st.session_state[fb_key] = ""
            st.session_state[clear_flag_key] = False

        # Now define text_area
        user_feedback = st.text_area(
            f"Feedback for Q{idx+1}",
            key=fb_key,
            placeholder="Enter feedback...",
        )

        # Button to update only this question
        if st.button(f"Update (Q{idx+1})"):
            if user_feedback.strip():
                refined = refine_answer(
                    question=qa["question_text"],
                    original_answer=qa["answer_text"],
                    user_feedback=user_feedback
                )
                st.session_state["answers"][idx]["answer_text"] = refined
                # Set a clear flag for next run
                st.session_state[clear_flag_key] = True
                st.success(f"Refined Q{idx+1} & cleared feedback.")
                st.rerun()
            else:
                st.info("No feedback provided, skipping update.")

    st.write("---")
    if st.button("Compile Final RFP"):
        final_text = compile_rfp_document(st.session_state["answers"])
        st.session_state["final_rfp_response"] = final_text
        st.success("Compiled final RFP. See below:")
        # st.write(final_text)

    if "final_rfp_response" in st.session_state:
        word_data = create_word_doc_from_text(st.session_state["final_rfp_response"])
        st.download_button(
            "Download as Word",
            data=word_data,
            file_name="my_rfp_response.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

def main():
    st.title("AI-Powered RFP Response with Chroma + BGE Embeddings")

    menu = st.sidebar.selectbox("Navigation", ["Home", "Index RFP Responses", "RFP Workflow"])
    
    if menu == "Home":
        st.write("""
            ## Welcome
            This example Streamlit application demonstrates how to:
            1. Ingest Company RFP responses into Chroma DB.
            2. Provide an RFP document (pasted text or uploaded).
            3. Automatically extract/generate RFP questions.
            4. Answer each question (with RAG from Chroma).
            5. Enable user feedback and finalize answers.
            6. Compile a final, detailed RFP Response document.

            Since "stop_at_node" is not supported in your current LangGraph version,
            we manually run each node in sequence.
        """)
    elif menu == "Index RFP Responses":
        page_index_existing_responses()
    elif menu == "RFP Workflow":
        page_run_workflow()

if __name__ == "__main__":
    main()
