import streamlit as st
from pathlib import Path
from graph_rag import GraphRAGAssistant, read_uploaded_file

st.set_page_config(page_title="GraphRAG Intelligence Assistant", page_icon="🕸️", layout="wide")
st.title("🕸️ GraphRAG Intelligence Assistant")
st.caption("Local-first document question answering using a knowledge graph + vector retrieval + an optional Ollama LLM.")

DATA_DIR = Path("data/sample_docs")
assistant = GraphRAGAssistant()

with st.sidebar:
    st.header("1. Add knowledge")
    uploaded = st.file_uploader("Upload TXT or PDF files", type=["txt", "pdf"], accept_multiple_files=True)
    if st.button("Index uploaded documents", type="primary", disabled=not uploaded):
        added = 0
        for file in uploaded:
            try:
                text, pages = read_uploaded_file(file)
                assistant.add_document(file.name, text, pages)
                added += 1
            except Exception as e:
                st.error(f"Could not read {file.name}: {e}")
        assistant.save()
        st.success(f"Indexed {added} document(s).")
    st.divider()
    st.header("2. Local LLM (optional)")
    ollama_model = st.text_input("Ollama model name", value="llama3.2:latest")
    use_llm = st.checkbox("Generate answers with Ollama", value=False)
    st.caption("Start Ollama separately and pull the model first. Without Ollama, the app returns retrieved evidence.")
    if st.button("Load sample documents"):
        count = assistant.load_sample_documents(DATA_DIR)
        assistant.save()
        st.success(f"Loaded {count} sample document(s).")

assistant.load_saved()
stats = assistant.stats()

m1, m2, m3 = st.columns(3)
m1.metric("Documents", stats["documents"])
m2.metric("Text chunks", stats["chunks"])
m3.metric("Graph entities", stats["entities"])

st.subheader("Ask a question")
question = st.text_input("Example: Why might the payment service fail after a database timeout?")
if st.button("Search and answer", type="primary") and question.strip():
    with st.spinner("Retrieving relevant chunks and graph connections..."):
        result = assistant.answer(question, use_llm=use_llm, model=ollama_model)
    st.markdown("### Answer")
    st.write(result["answer"])
    if result.get("warning"):
        st.info(result["warning"])

    left, right = st.columns(2)
    with left:
        st.markdown("### Retrieved evidence")
        if result["chunks"]:
            for item in result["chunks"]:
                with st.expander(f'{item["source"]} · relevance {item["score"]:.3f}'):
                    st.write(item["text"])
                    if item.get("page"):
                        st.caption(f'Page: {item["page"]}')
        else:
            st.write("No relevant passages found. Index documents or try a different question.")
    with right:
        st.markdown("### Related graph entities")
        if result["entities"]:
            for entity in result["entities"]:
                st.write(f'**{entity["name"]}** — {entity["relations"]}')
        else:
            st.write("No connected entities were identified for this query.")

st.divider()
st.subheader("Knowledge graph preview")
graph_data = assistant.graph_summary()
if graph_data["nodes"]:
    st.caption("Showing extracted entities and their relationships. This prototype uses NetworkX locally; Neo4j can be added as a production graph backend.")
    st.dataframe(graph_data["edges"], use_container_width=True, hide_index=True)
else:
    st.info("Load the sample documents or upload a document to build the graph.")

st.caption("Prototype note: entity extraction is heuristic. Verify extracted relationships before using the system for operational decisions.")
