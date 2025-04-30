import streamlit as st
import os
import tempfile
import warnings
import io
import gc
import PyPDF2
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# App title and configuration
st.set_page_config(page_title="Educational RAG App", page_icon="📚", layout="wide")
st.title("📚 Smart Study Assistant")
st.write("Upload your textbooks and get personalized in-depth learning content")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = "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"
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processing_errors' not in st.session_state:
    st.session_state.processing_errors = []
# Add temperature_override to session state
if 'temperature_override' not in st.session_state:
    st.session_state.temperature_override = 0.2
# Add max_tokens to session state
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 4000

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.success("API Key: ✓ Pre-configured")
    model_name = st.selectbox("Select Groq Model:", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"], index=0,
                             help="Large context window models provide more comprehensive answers")
    
    # PDF processing options
    st.header("PDF Processing Options")
    loader_option = st.selectbox(
        "PDF Loader Type:", 
        ["PyPDFLoader (Basic)", "PDFMinerLoader (Better text extraction)", "UnstructuredPDFLoader (Best for mixed content)"],
        index=1
    )
    batch_size = st.slider("Pages per batch:", min_value=1, max_value=20, value=5, 
                          help="Process this many pages at once. Lower values use less memory.")
    chunk_size = st.slider("Text chunk size:", min_value=200, max_value=2000, value=800,
                          help="Size of text chunks for processing. Increased for better context.")
    chunk_overlap = st.slider("Chunk overlap (%):", min_value=5, max_value=50, value=20,
                             help="Higher overlap improves context continuity between chunks.")
    skip_images = st.checkbox("Skip image extraction (faster processing)", value=True)
    
    st.header("Uploaded Textbooks")
    for file in st.session_state.uploaded_files:
        st.write(f"- {file}")
    
    st.header("Learning Preferences")
    learning_style = st.selectbox("Learning Style:", ["Visual", "Auditory", "Read/Write", "Kinesthetic", "Balanced"], index=4)
    complexity_level = st.select_slider("Content Complexity:", ["Beginner", "Intermediate", "Advanced", "Expert"], value="Intermediate")
    content_depth = st.select_slider("Answer Depth:", ["Basic", "Detailed", "Comprehensive", "Expert-level"], value="Comprehensive",
                                   help="Controls how in-depth the answers will be")
    include_examples = st.checkbox("Include examples in answers", value=True)
    include_analogies = st.checkbox("Include analogies in answers", value=True)
    include_questions = st.checkbox("Include practice questions", value=True)
    include_diagrams = st.checkbox("Suggest diagram structures", value=True)
    cross_reference = st.checkbox("Cross-reference information", value=True,
                                 help="Pull information from multiple sections to provide holistic answers")


def check_pdf(pdf_file):
    """Checks PDF properties before full processing."""
    # Check file size
    file_size_mb = len(pdf_file.getbuffer()) / (1024 * 1024)
    if file_size_mb > 100:  # If larger than 100MB
        return False, f"File too large ({file_size_mb:.1f}MB). Please use files under 100MB."
    
    # Check page count
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getbuffer()))
        page_count = len(pdf_reader.pages)
        if page_count > 500:  # If more than 500 pages
            return False, f"PDF has {page_count} pages. Consider splitting into smaller files (under 500 pages)."
        return True, f"PDF checks passed. {page_count} pages detected, {file_size_mb:.1f}MB in size."
    except Exception as e:
        return False, f"Error checking PDF: {str(e)}"


def process_large_pdf(pdf_path, loader_type="PDFMinerLoader", batch_size=5):
    """Process large PDFs in batches to avoid memory issues."""
    # Extract page count first
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total_pages = len(reader.pages)
    
    all_docs = []
    
    # Process in batches if using PyPDFLoader
    if loader_type == "PyPDFLoader (Basic)":
        for start_page in range(0, total_pages, batch_size):
            end_page = min(start_page + batch_size, total_pages) - 1
            loader = PyPDFLoader(pdf_path, pages=[i for i in range(start_page, end_page + 1)])
            try:
                docs = loader.load()
                all_docs.extend(docs)
                # Force garbage collection after each batch
                gc.collect()
            except Exception as e:
                st.warning(f"Error on pages {start_page}-{end_page}: {str(e)}")
    else:
        # For other loaders that don't support page ranges, load the whole document
        try:
            if loader_type == "PDFMinerLoader (Better text extraction)":
                loader = PDFMinerLoader(pdf_path)
            else:  # UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(pdf_path, mode="elements" if not skip_images else "single")
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            st.exception(e)
    
    return all_docs


def process_pdfs(pdf_files):
    """Processes uploaded PDFs and creates vector embeddings."""
    temp_dir = tempfile.mkdtemp()
    all_docs = []
    uploaded_filenames = []
    processing_errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Determine which loader to use based on sidebar selection
    loader_type = loader_option
    
    for i, pdf in enumerate(pdf_files):
        # Update progress
        progress_bar.progress((i) / len(pdf_files))
        status_text.text(f"Processing {pdf.name} ({i+1}/{len(pdf_files)})")
        
        # Check PDF before processing
        is_valid, message = check_pdf(pdf)
        if not is_valid:
            processing_errors.append(f"{pdf.name}: {message}")
            st.warning(message)
            continue
        
        # Save PDF to temporary file
        temp_pdf_path = os.path.join(temp_dir, pdf.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf.getbuffer())
        
        try:
            # Process the PDF in batches
            docs = process_large_pdf(temp_pdf_path, loader_type, batch_size)
            
            if docs:
                # Add filename metadata to each document for better source tracking
                for doc in docs:
                    if hasattr(doc, 'metadata'):
                        doc.metadata['source_file'] = pdf.name
                
                all_docs.extend(docs)
                uploaded_filenames.append(pdf.name)
                st.success(f"Successfully processed: {pdf.name}")
            else:
                message = f"No content extracted from: {pdf.name}"
                processing_errors.append(message)
                st.warning(message)
        except Exception as e:
            message = f"Error processing {pdf.name}: {str(e)}"
            processing_errors.append(message)
            st.error(message)
        
        # Force garbage collection after each file
        gc.collect()
    
    # Update progress to completion
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Create embeddings if documents were extracted
    if all_docs:
        status_text.text("Creating text chunks...")
        
        # Optimized text splitter with increased overlap for better context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * (chunk_overlap/100)),  # User-defined overlap percentage
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        try:
            splits = text_splitter.split_documents(all_docs)
            status_text.text(f"Creating embeddings for {len(splits)} text chunks...")
            
            # Using HuggingFace embeddings with scikit-learn
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Using SKLearnVectorStore
            vectorstore = SKLearnVectorStore.from_documents(
                documents=splits,
                embedding=embeddings
            )
            
            status_text.text("")  # Clear status
            st.success(f"✅ Created embeddings for {len(splits)} text chunks")
            return vectorstore, uploaded_filenames, processing_errors
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            st.exception(e)
            return None, [], processing_errors
    else:
        status_text.text("")  # Clear status
        return None, [], processing_errors


# Upload section
uploaded_pdfs = st.file_uploader("Upload PDF Textbooks", type="pdf", accept_multiple_files=True)

# Process button
if uploaded_pdfs and st.button("Process Textbooks", key="process_btn"):
    vectorstore, filenames, errors = process_pdfs(uploaded_pdfs)
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        # Only add unique filenames
        for filename in filenames:
            if filename not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(filename)
        st.session_state.processing_complete = True
        st.session_state.processing_errors = errors
        st.success("Textbooks processed and ready for questions!")
        
        # Show any errors that occurred
        if errors:
            with st.expander("Processing Issues"):
                for error in errors:
                    st.warning(error)


def create_rag_chain():
    """Creates an enhanced Retrieval-Augmented Generation (RAG) chain for in-depth answers."""
    if st.session_state.vectorstore is None:
        return None
    
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    
    # Enhanced retrieval configuration with error handling
    try:
        # Try with original parameters
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={
                "k": 8,  # Desired number of retrieved documents
                "score_threshold": 0.4,  # Lower threshold to get more relevant information
                "fetch_k": 15  # Fetch more candidates before filtering
            }
        )
    except ValueError as e:
        # If we get a ValueError (which happens when k is too large), fall back to safer values
        if "Expected n_neighbors" in str(e):
            st.warning("Limited document count detected. Using reduced retrieval parameters.")
            # Get a safer retriever with minimal parameters
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={
                    "k": 1,  # Minimum possible value
                    "score_threshold": 0.4,
                    "fetch_k": 2  # Minimum safe value
                }
            )
        else:
            # If it's a different error, re-raise it
            raise
    
    # Use temperature from session state - always get the current value
    llm = ChatGroq(
        model_name=model_name, 
        temperature=st.session_state.temperature_override,
        max_tokens=st.session_state.max_tokens
    )
    
    # Enhanced prompt template for more comprehensive answers
    template = """
    You are an expert educational assistant helping students learn effectively. Your goal is to provide COMPREHENSIVE, IN-DEPTH answers that thoroughly explore topics.
    
    Learning preferences:
    - Learning style: {learning_style}
    - Complexity level: {complexity_level}
    - Content depth: {content_depth}
    - Include examples: {include_examples}
    - Include analogies: {include_analogies}
    - Include practice questions: {include_questions}
    - Suggest diagram structures: {include_diagrams}
    - Cross-reference information: {cross_reference}
    
    Use the following context from textbooks to answer the question. This is the retrieved textbook content:
    
    {context}
    
    Question: {question}
    
    Important guidelines for your in-depth answer:
    1. Be COMPREHENSIVE - provide thorough explanations that cover all angles of the topic
    2. Provide DETAILED EXPLANATIONS of concepts, not just definitions
    3. Include MULTIPLE EXAMPLES with step-by-step explanations when helpful
    4. Make CONNECTIONS between different aspects of the topic
    5. Add PRACTICAL APPLICATIONS to demonstrate real-world relevance
    6. SYNTHESIZE information from different parts of the context when possible
    7. For complex topics, break down your explanation into clear sections with appropriate headings
    8. If warranted by complexity, include a summary section at the end
    9. When the content depth requested is "Comprehensive" or "Expert-level", don't hesitate to dive into advanced aspects
    10. If the source material contains technical terminology, properly define these terms
    11. For important concepts, explain BOTH what they are and why they matter

    If the context doesn't contain sufficient information, acknowledge this limitation but still provide the most comprehensive answer possible based on what's available.
    
    Format your response with appropriate headers, bullet points, and emphasis to enhance readability and understanding.
    
    Remember: Your goal is to create truly educational content that gives students a complete understanding of the topic.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Enhanced implementation of the RAG chain
    def get_context_and_question(question):
        with st.spinner("Searching through textbooks for comprehensive information..."):
            try:
                docs = retriever.invoke(question)
                
                # Extract source information for each document
                formatted_docs = []
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('source_file', 'Unknown source')
                    # Format each document with its source
                    content = doc.page_content.strip()
                    formatted_docs.append(f"[DOCUMENT {i+1} FROM {source}]:\n{content}\n")
                
                # Join all formatted documents
                context = "\n\n".join(formatted_docs)
                
                if not formatted_docs:
                    context = "No relevant information found in the textbooks. Please provide a general answer based on your knowledge."
                    st.warning("No relevant information found in the textbooks for this question.")
                
            except Exception as e:
                st.error(f"Error retrieving information: {str(e)}")
                context = "An error occurred while searching the textbooks. Please provide a general answer based on your knowledge."
        
        return {
            "context": context,
            "question": question,
            "learning_style": learning_style,
            "complexity_level": complexity_level,
            "content_depth": content_depth,
            "include_examples": "Yes" if include_examples else "No",
            "include_analogies": "Yes" if include_analogies else "No",
            "include_questions": "Yes" if include_questions else "No",
            "include_diagrams": "Yes" if include_diagrams else "No",
            "cross_reference": "Yes" if cross_reference else "No"
        }
    
    rag_chain = (
        get_context_and_question
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def create_study_guide_chain():
    """Creates a specialized RAG chain for comprehensive study guides."""
    if st.session_state.vectorstore is None:
        return None
    
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    
    # Special retriever configuration for study guides with error handling
    try:
        # Try with original parameters
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={
                "k": 12,  # More documents for comprehensive coverage
                "score_threshold": 0.25,  # Lower threshold to capture more related content
                "fetch_k": 25  # Even more candidates to ensure topic breadth
            }
        )
    except ValueError as e:
        # If we get a ValueError (which happens when k is too large), fall back to safer values
        if "Expected n_neighbors" in str(e):
            st.warning("Limited document count detected. Using reduced retrieval parameters for study guide.")
            # Get a safer retriever with minimal parameters
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={
                    "k": 1,  # Minimum possible value
                    "score_threshold": 0.3,
                    "fetch_k": 2  # Minimum safe value
                }
            )
        else:
            # If it's a different error, re-raise it
            raise
    
    # Use temperature from session state - always get the current value
    # Use a slightly higher temperature for creativity in study guide generation
    llm = ChatGroq(
        model_name=model_name, 
        temperature=max(0.3, st.session_state.temperature_override),  # Minimum 0.3 for study guides
        max_tokens=min(8000, st.session_state.max_tokens * 2)  # Double the tokens but cap at 8000
    )
    
    # Improved prompt template for study guides that focuses on content completion
    template = """
    You are an expert educational content creator specializing in comprehensive study guides. Your task is to create a COMPLETE, IN-DEPTH study guide that thoroughly covers the requested topic.
    
    Learning preferences:
    - Learning style: {learning_style}
    - Complexity level: {complexity_level}
    - Content depth: {content_depth}
    - Guide format: {guide_format}
    - Include examples: {include_examples}
    - Include analogies: {include_analogies}
    - Include practice questions: {include_questions}
    - Suggest diagram structures: {include_diagrams}
    
    Use the following context from textbooks to create a study guide. This is the retrieved textbook content:
    
    {context}
    
    Topic: {topic}
    Number of main sections requested: {num_sections}
    
    IMPORTANT: You MUST create a COMPLETE study guide with FULL CONTENT for each section, not just outlines or section titles. Each section must have several paragraphs of detailed information.
    
    Your study guide MUST include:
    
    # {topic} - Comprehensive Study Guide
    
    ## Introduction
    [Write 2-3 paragraphs introducing the topic, its importance, and what students will learn]
    
    ## Section 1: [First Main Topic]
    [Write 3-5 paragraphs with detailed information about this section]
    [Include specific definitions, explanations, and examples]
    [Add relevant subsections with their own content]
    
    ## Section 2: [Second Main Topic]
    [Write 3-5 paragraphs with detailed information about this section]
    [Include specific definitions, explanations, and examples]
    [Add relevant subsections with their own content]
    
    [Continue with remaining sections...]
    
    ## Practice Questions
    1. [First question with detailed answer]
    2. [Second question with detailed answer]
    [Include at least 3-5 practice questions with comprehensive answers]
    
    ## Summary
    [Write 1-2 paragraphs summarizing the key points of the entire topic]
    
    Remember: You MUST provide COMPLETE, DETAILED CONTENT for each section, not just outlines or headings. Your study guide should be comprehensive and ready for students to use immediately.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Enhanced implementation for study guide generation with improved context gathering
    def get_context_and_topic(topic):
        # Create multiple targeted queries for better coverage
        targeted_queries = [
            f"{topic} definition and explanation",
            f"{topic} key concepts and principles",
            f"{topic} examples and applications",
            f"{topic} important details",
            f"{topic} related concepts",
            f"information about {topic}"
        ]
        
        with st.spinner("Gathering comprehensive information for your study guide..."):
            st.text("Step 1/4: Retrieving relevant content from your textbooks...")
            
            all_docs = []
            # Run each targeted query
            for query in targeted_queries:
                try:
                    docs = retriever.invoke(query)
                    all_docs.extend(docs)
                except Exception as e:
                    st.warning(f"Error retrieving information for '{query}': {str(e)}")
                    continue
            
            # Also do a direct search for the topic itself
            try:
                docs = retriever.invoke(topic)
                all_docs.extend(docs)
            except Exception as e:
                st.warning(f"Error retrieving information for the topic directly: {str(e)}")
            
            # Check if we found any documents
            if not all_docs:
                st.warning("No relevant information found in the textbooks for this topic.")
                context = "No relevant information found in the textbooks. Please create a study guide based on your general knowledge of the topic."
                st.text("Step 2/4: Generating study guide with general knowledge...")
                return {
                    "context": context,
                    "topic": topic,
                    "num_sections": num_sections,
                    "guide_format": guide_format,
                    "learning_style": learning_style,
                    "complexity_level": complexity_level,
                    "content_depth": content_depth,
                    "include_examples": "Yes" if include_examples else "No",
                    "include_analogies": "Yes" if include_analogies else "No",
                    "include_questions": "Yes" if include_questions else "No",
                    "include_diagrams": "Yes" if include_diagrams else "No"
                }
            
            # Deduplicate documents by content to avoid repetition
            st.text("Step 2/4: Deduplicating and processing content...")
            unique_content = {}
            unique_docs = []
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
                if content_hash not in unique_content:
                    unique_content[content_hash] = True
                    unique_docs.append(doc)
            
            # Sort documents by relevance (if available in metadata)
            try:
                unique_docs.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
            except:
                pass  # Skip sorting if not possible
            
            st.text(f"Step 3/4: Processing {len(unique_docs)} relevant sections...")
            
            # Extract source information for each document
            formatted_docs = []
            for i, doc in enumerate(unique_docs):
                source = doc.metadata.get('source_file', 'Unknown source')
                content = doc.page_content.strip()
                formatted_docs.append(f"[DOCUMENT {i+1} FROM {source}]:\n{content}\n")
            
            # Join all formatted documents with clear separation
            context = "\n\n" + "\n\n".join(formatted_docs) + "\n\n"
            
            # Add a final preprocessing step to analyze the context
            st.text("Step 4/4: Analyzing content and generating study guide...")
        
        return {
            "context": context,
            "topic": topic,
            "num_sections": num_sections,
            "guide_format": guide_format,
            "learning_style": learning_style,
            "complexity_level": complexity_level,
            "content_depth": content_depth,
            "include_examples": "Yes" if include_examples else "No",
            "include_analogies": "Yes" if include_analogies else "No",
            "include_questions": "Yes" if include_questions else "No",
            "include_diagrams": "Yes" if include_diagrams else "No"
        }
    
    study_guide_chain = (
        get_context_and_topic
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return study_guide_chain


st.header("Learn From Your Textbooks")
tab1, tab2, tab3 = st.tabs(["Ask Questions", "Study Guide Generator", "System Status"])

with tab1:
    # Always show the text input, but disable it if vectorstore is None
    question = st.text_input(
        "What would you like to learn about?", 
        key="question_input", 
        disabled=(st.session_state.vectorstore is None),
        placeholder="Ask a specific question for a comprehensive answer..."
    )
    
    # Always show the button, but disable it if vectorstore is None or question is empty
    if st.button(
        "Get In-Depth Answer", 
        key="answer_btn", 
        disabled=(st.session_state.vectorstore is None or not question)
    ):
        try:
            rag_chain = create_rag_chain()
            if rag_chain:
                with st.spinner("Creating a comprehensive answer for you..."):
                    response = rag_chain.invoke(question)
                st.markdown(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    # Show info message if vectorstore is None
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks before asking questions.")

with tab2:
    # Always show the topic input, but disable it if vectorstore is None
    topic = st.text_input(
        "Topic for comprehensive study guide:", 
        key="topic_input", 
        disabled=(st.session_state.vectorstore is None),
        placeholder="Enter a specific topic for your in-depth study guide..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_sections = st.slider("Number of main sections:", min_value=3, max_value=10, value=5, 
                                help="More sections allow for more detailed organization")
    
    with col2:
        guide_format = st.selectbox(
            "Study guide format:",
            ["Comprehensive", "Condensed", "Visual-focused", "Practice-oriented"],
            index=0
        )
    
    # Always show the button, but disable it if vectorstore is None or topic is empty
    if st.button(
        "Generate Comprehensive Study Guide", 
        key="guide_btn", 
        disabled=(st.session_state.vectorstore is None or not topic)
    ):
        try:
            study_guide_chain = create_study_guide_chain()
            if study_guide_chain:
                with st.spinner("Creating your in-depth study guide... This may take a minute for comprehensive content."):
                    response = study_guide_chain.invoke(topic)
                st.markdown(response)
                
                # Add option to download the study guide as markdown
                from io import BytesIO
                
                # Create a download button for the study guide
                buffer = BytesIO()
                buffer.write(response.encode())
                buffer.seek(0)
                
                st.download_button(
                    label="Download Study Guide (Markdown)",
                    data=buffer,
                    file_name=f"{topic.replace(' ', '_')}_study_guide.md",
                    mime="text/markdown"
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    # Show info message if vectorstore is None
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks first.")

with tab3:
    st.subheader("System Status")
    
    # Memory usage information
    import psutil
    memory = psutil.virtual_memory()
    st.metric("Memory Usage", f"{memory.percent}%", f"{memory.available / (1024 * 1024 * 1024):.2f} GB free")
    
    # Textbook status
    if st.session_state.uploaded_files:
        st.success(f"{len(st.session_state.uploaded_files)} textbooks processed")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file}")
    else:
        st.warning("No textbooks processed yet")
    
    # Processing errors
    if st.session_state.processing_errors:
        with st.expander("Processing Issues"):
            for error in st.session_state.processing_errors:
                st.warning(error)
    
    # Vector store info
    if st.session_state.vectorstore:
        try:
            st.success("Vector store is active and ready for in-depth queries")
        except:
            st.error("Vector store exists but may have issues")
    else:
        st.warning("No vector store available")
    
    # Clear temporary files button
    if st.button("Clear Temporary Files"):
        import shutil
        try:
            temp_dir = tempfile.gettempdir()
            # Clean temporary directories that start with tmp
            for item in os.listdir(temp_dir):
                if item.startswith('tmp'):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
            st.success("Temporary files cleared successfully")
        except Exception as e:
            st.error(f"Error clearing temporary files: {str(e)}")

# Advanced settings expander
with st.expander("Advanced Settings"):
    st.subheader("Response Generation Settings")
    
    # Store previous values to detect changes
    prev_max_tokens = st.session_state.max_tokens
    prev_temperature = st.session_state.temperature_override
    
    # Update max_tokens in session state
    st.session_state.max_tokens = st.slider("Maximum response length:", min_value=500, max_value=10000, value=st.session_state.max_tokens,
                       help="Maximum number of tokens in generated responses")
    
    # Update temperature_override in session state
    st.session_state.temperature_override = st.slider("Temperature override:", min_value=0.0, max_value=1.0, value=st.session_state.temperature_override, step=0.05,
                                 help="Lower values create more focused, deterministic responses")
    
    # Display indicators when settings have been changed
    if st.session_state.temperature_override != 0.2:  # Default is 0.2
        st.info(f"Temperature set to {st.session_state.temperature_override} (default: 0.2). Higher values produce more creative responses.")
    
    if st.session_state.max_tokens != 4000:  # Default is 4000
        st.info(f"Maximum response length set to {st.session_state.max_tokens} tokens (default: 4000).")
    
    st.info("Note: These settings will apply to newly generated responses.")
    
    # Add a reset button for advanced settings
    if st.button("Reset to Default Settings"):
        st.session_state.temperature_override = 0.2  # Default temperature
        st.session_state.max_tokens = 4000  # Default max tokens
        st.experimental_rerun()

# Reset button at the bottom
if st.button("Reset App", key="reset_btn"):
    # Clean up session state
    st.session_state.vectorstore = None
    st.session_state.uploaded_files = []
    st.session_state.processing_complete = False
    st.session_state.processing_errors = []
    
    # Force garbage collection
    gc.collect()
    
    st.success("App reset successfully. You can upload new textbooks now.")
    st.experimental_rerun()
