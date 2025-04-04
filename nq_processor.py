# pip install datasets
# pip install nltk
# pip install beautifulsoup4

# Handle NLTK download issues

try:
    import ssl
    import nltk

    # Try creating an unverified SSL context
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Try downloading punkt
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except Exception as e:
    print(f"Warning: NLTK data download failed: {str(e)}")
    print("Continuing execution, but some text processing features might be limited...")

from datasets import load_dataset
import os
import hashlib
import pandas as pd
import random
import time
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def is_table_content(text):
    """Detect if the text is table content"""
    table_indicators = ['<table>', '</table>', '<tr>', '</tr>', '<td>', '</td>', '<th>', '</th>']
    structural_indicators = ['No.', 'Episode', 'Title', 'Directed by', 'Written by', 'Original air date']

    # Check for HTML table tags
    for indicator in table_indicators:
        if indicator in text:
            return True

    # Check for structural indicators of table content
    matches = 0
    for indicator in structural_indicators:
        if indicator in text:
            matches += 1

    # If multiple structural indicators are present, it might be table data
    return matches >= 3


def extract_meaningful_paragraph(tokens, min_length=100):
    """Extract a meaningful paragraph from tokens"""
    paragraphs = []
    current_para = []
    in_paragraph = False

    # First, try using HTML paragraph tags
    for i, token in enumerate(tokens):
        if token == '<P>':
            in_paragraph = True
            current_para = []
        elif token == '</P>':
            if current_para:
                paragraph_text = ' '.join(current_para)
                # Clean HTML
                paragraph_text = re.sub(r'<[^>]+>', ' ', paragraph_text)
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text).strip()
                if len(paragraph_text) >= min_length:
                    paragraphs.append(paragraph_text)
            in_paragraph = False
            current_para = []
        elif in_paragraph:
            current_para.append(token)

    # If no paragraph tags were found, try splitting based on sentences
    if not paragraphs:
        # Merge all tokens
        text = ' '.join(tokens)
        # Clean HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Use NLTK to split sentences
        sentences = nltk.sent_tokenize(text)

        # Merge sentences into paragraphs (every 4-5 sentences)
        for i in range(0, len(sentences), 4):
            paragraph = ' '.join(sentences[i:i + 4])
            if len(paragraph) >= min_length:
                paragraphs.append(paragraph)

    # Return the longest paragraph (or an empty string)
    return max(paragraphs, key=len, default="") if paragraphs else ""


def extract_answer_text(example):
    """Improved method to extract answer text from the NQ dataset"""
    # First, try to get the long answer
    extracted_answer = ""

    # Ensure tokens exist
    if 'document' not in example or 'tokens' not in example['document']:
        return "Could not extract answer text"

    tokens = example['document']['tokens'].get('token', [])
    if not tokens:
        return "Could not extract answer text"

    # Try getting the long answer from annotations
    if 'annotations' in example and 'long_answer' in example['annotations']:
        long_answers = example['annotations']['long_answer']
        if isinstance(long_answers, list) and long_answers and 'start_token' in long_answers[0]:
            start_token = long_answers[0].get('start_token')
            end_token = long_answers[0].get('end_token')

            if start_token is not None and end_token is not None and 0 <= start_token < end_token < len(tokens):
                # Extract answer tokens
                answer_tokens = tokens[start_token:end_token]
                extracted_text = ' '.join(answer_tokens)

                # Check if it is table content
                if is_table_content(extracted_text):
                    # It's a table, try extracting a paragraph
                    extracted_answer = extract_meaningful_paragraph(tokens)
                else:
                    # Clean HTML tags
                    extracted_answer = re.sub(r'<[^>]+>', ' ', extracted_text)
                    extracted_answer = re.sub(r'\s+', ' ', extracted_answer).strip()

    # If no valid answer is found, try using relevant paragraphs from the entire document
    if not extracted_answer or len(extracted_answer) < 100:
        # Try finding relevant paragraphs using question keywords
        if 'question' in example and 'text' in example['question']:
            question = example['question']['text']
            # Extract keywords from the question
            keywords = [word.lower() for word in re.findall(r'\b\w+\b', question)
                        if
                        len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'who', 'how', 'does',
                                                               'did', 'this', 'that', 'these', 'those', 'have', 'from']]

            # Merge tokens into text and split into paragraphs
            full_text = ' '.join(tokens)
            # Clean HTML
            full_text = re.sub(r'<[^>]+>', ' ', full_text)
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Split into paragraphs
            paragraphs = [p for p in full_text.split('.') if len(p) > 50]

            # Score each paragraph (based on keyword matching)
            paragraph_scores = []
            for para in paragraphs:
                score = 0
                para_lower = para.lower()
                for keyword in keywords:
                    if keyword in para_lower:
                        score += 1
                paragraph_scores.append((para, score))

            # Select the highest-scoring paragraph
            paragraph_scores.sort(key=lambda x: x[1], reverse=True)
            if paragraph_scores and paragraph_scores[0][1] > 0:
                extracted_answer = paragraph_scores[0][0].strip()

    # Final check - if still no suitable answer found, use a long paragraph from the document
    if not extracted_answer or len(extracted_answer) < 100:
        extracted_answer = extract_meaningful_paragraph(tokens)

    # If still no answer is found, return a standard message
    if not extracted_answer or len(extracted_answer) < 50:
        return "Could not extract a suitable answer based on the provided document content"

    return extracted_answer


def clean_html_content(text):
    """Clean HTML content and try to preserve text structure"""
    # Use BeautifulSoup to handle potential HTML fragments
    try:
        # Add a root tag to handle potentially incomplete HTML
        soup = BeautifulSoup(f"<root>{text}</root>", 'html.parser')
        # Get all text
        text = soup.get_text(' ', strip=True)
    except:
        # If BeautifulSoup processing fails, use regular expressions
        text = re.sub(r'<[^>]+>', ' ', text)

    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_nq_for_rag(dataset, output_path="nq_for_rag.csv", sample_size=50):
    """Process the NQ dataset into a format suitable for RAG evaluation"""
    processed_data = []

    print(f"Starting processing of Natural Questions dataset, target sample size: {sample_size}")

    # Process the dataset
    sample_count = min(len(dataset), sample_size * 3)  # Process a bit more to ensure enough samples can be filtered

    for i in range(sample_count):
        if i % 10 == 0:
            print(f"Processed {i}/{sample_count} examples...")

        example = dataset[i]

        try:
            # Get question text
            question_text = ""
            if isinstance(example['question'], dict) and 'text' in example['question']:
                question_text = example['question']['text']
            else:
                continue  # Skip examples without valid questions

            # Get document title
            document_title = ""
            if isinstance(example['document'], dict) and 'title' in example['document']:
                document_title = example['document']['title']
            else:
                document_title = f"document_{i}"

            # Extract answer text
            answer_text = extract_answer_text(example)

            # Check if the answer is clearly table content or poorly formatted
            if is_table_content(answer_text) or answer_text.startswith("Could not extract a suitable answer"):
                continue

            # Ensure the question and answer have sufficient quality
            if len(question_text) > 5 and len(answer_text) > 100:
                # Check question quality - avoid URLs or incomplete questions
                if "http" in question_text or question_text.count(" ") < 2:
                    continue

                # Ensure the answer is not a remnant of tables or structured data
                if any(marker in answer_text for marker in
                       ['No.', 'Episode', 'Title', 'Directed by', 'Original air date']):
                    if answer_text.count('|') > 3 or answer_text.count(':') > 5:
                        continue

                processed_data.append({
                    "question": question_text,
                    "document_id": document_title,
                    "answer_1": answer_text[:1000],  # Limit length
                    "answer_2": "",
                    "answer_3": ""
                })

                # If enough samples have been collected, can finish early
                if len(processed_data) >= sample_size:
                    break
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            continue

    # If more samples are collected than the target number, select randomly
    if len(processed_data) > sample_size:
        processed_data = random.sample(processed_data, sample_size)

    # Check quality and filter out unqualified samples
    filtered_data = []
    for item in processed_data:
        # Further quality checks
        if len(item["answer_1"]) > 100 and not is_table_content(item["answer_1"]):
            filtered_data.append(item)

    # Save as CSV
    df = pd.DataFrame(filtered_data)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} NQ examples to {output_path}")

    return df


def extract_document_text(example):
    """Improved method to extract complete document text from NQ examples"""
    if 'document' in example and 'tokens' in example['document'] and 'token' in example['document']['tokens']:
        tokens = example['document']['tokens']['token']

        # Try segmenting based on HTML tags
        paragraphs = []
        current = []
        in_title = False
        in_paragraph = False
        in_table = False
        title = ""

        for token in tokens:
            # Handle title
            if token == '<title>':
                in_title = True
                current = []
            elif token == '</title>':
                in_title = False
                title = ' '.join(current)
                current = []
            # Handle paragraph
            elif token == '<P>' or token == '<p>':
                in_paragraph = True
                current = []
            elif token == '</P>' or token == '</p>':
                if current:
                    # Keep only non-table content paragraphs
                    paragraph_text = ' '.join(current)
                    if not is_table_content(paragraph_text):
                        # Clean HTML tags
                        cleaned = re.sub(r'<[^>]+>', ' ', paragraph_text)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        if len(cleaned) > 50:  # Keep only meaningful paragraphs
                            paragraphs.append(cleaned)
                in_paragraph = False
                current = []
            # Skip table content
            elif token == '<table>':
                in_table = True
                current = []
            elif token == '</table>':
                in_table = False
                current = []
            # Collect current content
            elif (in_title or in_paragraph) and not in_table:
                current.append(token)

        # If not enough paragraphs found, try splitting based on periods
        if len(paragraphs) < 3:
            text = ' '.join(tokens)
            # Clean HTML
            text = clean_html_content(text)
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            # Reassemble into paragraphs (every 5 sentences)
            for i in range(0, len(sentences), 5):
                paragraph = ' '.join(sentences[i:i + 5])
                if len(paragraph) > 50:
                    paragraphs.append(paragraph)

        # Build the final document
        document_text = ""
        if title:
            document_text += f"Title: {clean_html_content(title)}\n\n" # Translated "标题"

        document_text += "\n\n".join(paragraphs)

        return document_text

    return ""


def prepare_nq_documents(dataset, output_folder="nq_documents"):
    """Improved method to extract documents from the NQ dataset as a knowledge base"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a mapping from document ID to file path to avoid duplicate processing
    doc_id_map = {}

    doc_count = 0
    processed_count = 0

    print("Starting document extraction...")

    # Extract documents from the datase
    for i in range(len(dataset)):  # Traverse all 3,500 samples loaded
        try:
            example = dataset[i]

            # Get document title
            document_title = ""
            if isinstance(example['document'], dict) and 'title' in example['document']:
                document_title = example['document']['title']
            else:
                document_title = f"document_{i}"

            doc_id = hashlib.md5(document_title.encode()).hexdigest()

            # If this document has already been processed, skip it
            if doc_id in doc_id_map:
                continue

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} examples, saved {doc_count} documents")

            # Extract document content
            text = extract_document_text(example)

            # Check document quality - avoid tables or too short documents
            if not text or len(text) < 300 or is_table_content(text):
                continue

            # Use the title as the filename (hashed to avoid illegal characters)
            file_name = doc_id + ".txt"
            file_path = os.path.join(output_folder, file_name)
            doc_id_map[doc_id] = file_path

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            doc_count += 1

            # Limit the number of documents processed
            if doc_count >= 300:  # Reduce document count for efficiency
                break
        except Exception as e:
            print(f"Error processing document {i}: {str(e)}")
            continue

    print(f"Saved a total of {doc_count} documents to the {output_folder} directory")
    return doc_id_map


def create_manual_test_samples(output_path="manual_rag_test.csv"):
    """Create some manually defined high-quality test samples"""
    samples = [
        # --- Samples unchanged as they were already in English ---
        {
            "question": "What are the main components of a RAG system?",
            "answer_1": "The main components of a Retrieval-Augmented Generation (RAG) system include: 1) A vector database or retrieval system that stores and indexes documents or passages from a knowledge base, 2) An embedding model that converts text into vector representations, 3) A retrieval mechanism that finds relevant information based on query similarity, 4) A large language model (LLM) that generates responses, and 5) A prompt engineering system that combines retrieved context with user queries to guide the LLM's response generation. These components work together to enhance the accuracy and factual grounding of AI-generated responses by leveraging external knowledge sources.",
            "document_id": "manual_doc_1"
        },
        {
            "question": "What are the benefits of using a RAG system over a standard LLM?",
            "answer_1": "RAG systems offer several advantages over standard LLMs: 1) Improved factual accuracy by grounding responses in reliable sources, 2) Reduced hallucinations since the model references external knowledge rather than relying solely on parametric memory, 3) Better handling of domain-specific queries by incorporating specialized knowledge bases, 4) Ability to access up-to-date information beyond the model's training cutoff, 5) Greater transparency as sources can be cited, 6) Enhanced control over response generation, and 7) Lower computational requirements compared to constantly retraining or fine-tuning LLMs with new information. These benefits make RAG particularly valuable for applications requiring high factual accuracy and domain expertise.",
            "document_id": "manual_doc_2"
        },
        {
            "question": "What metrics are used to evaluate RAG systems?",
            "answer_1": "RAG systems are evaluated using several specialized metrics: 1) Retrieval-focused metrics like precision, recall, and Mean Average Precision (MAP) that assess the quality of retrieved documents; 2) Generation quality metrics including ROUGE, BLEU, and BERTScore that measure the linguistic quality of generated responses; 3) Factual accuracy metrics such as knowledge F1 score and faithfulness that evaluate whether the generated content aligns with retrieved information; 4) Context relevance metrics that assess if the retrieved context contains information needed to answer the query; 5) Human evaluation scales for coherence, relevance, and helpfulness; and 6) Efficiency metrics like latency and computational cost. Comprehensive evaluation typically involves a combination of these automatic metrics and human assessment.",
            "document_id": "manual_doc_3"
        },
        {
            "question": "How does vector similarity search work in RAG?",
            "answer_1": "Vector similarity search in RAG works by converting text into numerical vector representations (embeddings) and finding content with similar vectors. The process involves: 1) Embedding generation - text is transformed into high-dimensional vectors using models like BERT or Sentence Transformers; 2) Vector storage - these embeddings are stored in specialized vector databases like FAISS or Pinecone; 3) Similarity computation - when a query arrives, it's also embedded, and the system calculates how similar it is to stored vectors using metrics like cosine similarity or Euclidean distance; 4) Efficient retrieval - various algorithms (approximate nearest neighbor search, clustering, indexing) enable fast retrieval even with millions of vectors; 5) Ranking and selection - the most similar documents are ranked and selected for context augmentation. This process enables RAG systems to quickly identify contextually relevant information from large knowledge bases.",
            "document_id": "manual_doc_4"
        },
        {
            "question": "What are the challenges in implementing an effective RAG system?",
            "answer_1": "Implementing effective RAG systems involves several key challenges: 1) Context length limitations - balancing comprehensive context with model token limits; 2) Retrieval quality issues - ensuring relevant information is retrieved while filtering out noise; 3) Semantic search limitations - difficulty capturing nuanced relationships beyond keyword matching; 4) Computational efficiency concerns - managing latency with large knowledge bases; 5) Data freshness management - keeping information updated without constant reprocessing; 6) Cross-lingual and multimodal challenges - handling diverse languages and content types; 7) Hallucination mitigation - preventing the LLM from generating facts not in the retrieved context; 8) Effective chunking strategies - determining optimal document segmentation; 9) Evaluation complexity - developing reliable metrics beyond simple retrieval or generation metrics; and 10) Balancing retrieval depth with response generation quality. Addressing these challenges requires careful system design and continual refinement.",
            "document_id": "manual_doc_5"
        }
    ]

    # Save as CSV
    df = pd.DataFrame(samples)
    df['answer_2'] = ""
    df['answer_3'] = ""
    df.to_csv(output_path, index=False)
    print(f"Saved {len(samples)} manually created test samples to {output_path}")

    return df


def create_manual_documents(output_folder="manual_documents"):
    """Create manually defined high-quality documents as a knowledge base"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    documents = {
        # --- Documents unchanged as they were already in English ---
        "rag_components.txt": """
Retrieval-Augmented Generation (RAG) Systems: Core Components and Architecture

A RAG system integrates retrieval mechanisms with generative AI to produce more accurate, factual responses. Here are the essential components that make up a modern RAG architecture:

1. Knowledge Base
The foundation of any RAG system is its knowledge base - a collection of documents, passages, or structured data that contains the information the system can access. This can include:
- Documents (PDFs, web pages, articles)
- Structured databases
- API-accessible information sources
- Internal company documents
- Public knowledge repositories

2. Document Processing Pipeline
Before retrieval, documents must be processed through several steps:
- Chunking: Dividing documents into smaller, manageable pieces
- Cleaning: Removing irrelevant formatting or content
- Metadata extraction: Identifying key attributes like date, author, source
- Entity recognition: Identifying important names, organizations, or concepts

3. Embedding Models
These models convert text into vector representations (embeddings) that capture semantic meaning:
- Document embeddings: Vector representations of document chunks
- Query embeddings: Vector representations of user questions
- Common embedding models include Sentence Transformers, OpenAI embeddings, or custom domain-specific embeddings

4. Vector Database
Specialized databases optimized for storing and retrieving vector embeddings:
- FAISS (Facebook AI Similarity Search)
- Pinecone
- Weaviate
- Chroma
- Milvus
These databases support efficient similarity search across millions or billions of vectors.

5. Retrieval System
The component that identifies and retrieves relevant information based on the query:
- Vector similarity search (nearest neighbors)
- Hybrid retrieval (combining vector search with keyword or BM25)
- Re-ranking mechanisms to improve precision
- Filtering capabilities based on metadata

6. Large Language Model (LLM)
The generative AI component that produces responses:
- Can be open-source (Llama, Mistral, MPT) or proprietary (GPT, Claude)
- Handles the generation based on context and query
- May need to be fine-tuned for specific RAG applications

7. Prompt Engineering System
Creates effective prompts by combining:
- Retrieved context from the knowledge base
- The original user query
- System instructions and guidelines
- Examples or few-shot demonstrations when needed

8. Orchestration Layer
Coordinates the entire RAG workflow:
- Manages the pipeline from query to response
- Handles error cases and fallbacks
- Controls caching and optimization
- Monitors system performance

9. Evaluation Framework
Tools and metrics to assess RAG performance:
- Retrieval quality metrics (precision, recall, etc.)
- Generation quality metrics
- Factual accuracy assessment
- Latency and efficiency metrics

When designed properly, these components work together to create a system that leverages both the factual knowledge from external sources and the linguistic capabilities of large language models.
        """,

        "rag_benefits.txt": """
Benefits of RAG Systems Over Standard LLMs

Retrieval-Augmented Generation (RAG) systems offer significant advantages compared to using standard large language models (LLMs) alone. Here's a comprehensive analysis of these benefits:

Enhanced Factual Accuracy
Standard LLMs rely entirely on information encoded in their parameters during training, leading to potential inaccuracies when recalling specific facts. RAG systems directly access external knowledge bases at inference time, dramatically improving factual accuracy by grounding responses in reliable sources.

Reduced Hallucinations
One of the most significant limitations of standard LLMs is their tendency to "hallucinate" or generate plausible-sounding but incorrect information. By retrieving and referencing actual documents, RAG systems substantially reduce hallucinations, as the model can explicitly ground its responses in the retrieved context.

Domain Specialization Without Fine-tuning
Adapting a standard LLM to specialized domains typically requires expensive fine-tuning or retraining. RAG systems can instantly become domain experts by simply connecting to domain-specific knowledge bases, whether medical literature, legal documents, or technical manuals, without modifying the underlying model weights.

Access to Up-to-date Information
Standard LLMs have knowledge cutoffs beyond which they have no information. RAG systems can access continuously updated knowledge bases, allowing them to reference current events, recent research, or the latest documentation without retraining the model.

Transparency and Attributability
RAG systems can cite their sources directly from the retrieved documents, providing clear evidence for their claims. This transparency helps users verify information and understand where answers come from, unlike standard LLMs which typically cannot provide specific sources for their outputs.

Knowledge Base Control
Organizations using RAG have precise control over what information sources the system can access. This enables filtering out unreliable sources and ensuring compliance with organizational guidelines or industry regulations.

Computational Efficiency
Continuously retraining large models to incorporate new knowledge is computationally expensive. RAG systems separate knowledge storage from reasoning capabilities, making it much more efficient to update information by simply modifying the knowledge base rather than retraining the model.

Customization Without Model Access
RAG allows customizing responses using proprietary data without requiring direct access to modify the underlying LLM. This is particularly valuable when using commercially available models where direct fine-tuning might not be possible.

Improved Long-tail Performance
Standard LLMs often struggle with niche or specialized questions outside common knowledge. RAG systems excel at handling these "long-tail" queries by retrieving specific information that might be too obscure to be reliably encoded in model parameters.

Mitigation of Training Data Biases
RAG can help mitigate biases present in an LLM's training data by providing alternative viewpoints from curated sources, effectively supplementing the model's parametric knowledge with more balanced external information.

These advantages make RAG particularly valuable for applications requiring high factual accuracy, up-to-date information, and domain expertise, such as customer support, research assistance, educational tools, and enterprise knowledge systems.
        """,

        "rag_evaluation.txt": """
Comprehensive Metrics for Evaluating RAG Systems

Evaluating Retrieval-Augmented Generation (RAG) systems requires assessing both retrieval quality and generation capabilities. Here's a detailed overview of the metrics used to evaluate different aspects of RAG performance:

Retrieval-Focused Metrics

Precision@K: Measures the proportion of relevant documents among the top K retrieved documents. Higher values indicate the retrieval system is returning mostly relevant information.

Recall@K: Evaluates the proportion of all relevant documents that appear in the top K retrieved results. This measures how comprehensively the system captures available relevant information.

Mean Average Precision (MAP): Calculates the mean of average precision scores for each query, providing a single-figure measure of quality across recall levels.

Normalized Discounted Cumulative Gain (nDCG): Measures retrieval quality while accounting for the position of relevant documents in the results list, with higher weights for higher-ranked positions.

Mean Reciprocal Rank (MRR): The average of the reciprocal ranks of the first relevant document for a set of queries. Focuses on how quickly the system finds at least one relevant document.

Generation Quality Metrics

ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures overlap of n-grams between the generated response and reference answers. ROUGE-1, ROUGE-2, and ROUGE-L are common variants measuring unigram, bigram, and longest common subsequence matches respectively.

BLEU (Bilingual Evaluation Understudy): Evaluates the precision of n-gram matches between generated text and references, with penalties for overly short outputs.

BERTScore: Uses contextual embeddings to compute similarity between generated and reference texts, capturing semantic similarity beyond exact word matches.

Factual Accuracy Metrics

Knowledge F1: Measures how well the generated response captures key facts from the reference answer.

Faithfulness: Evaluates whether the generated content is supported by the retrieved context, penalizing hallucinations or contradictions.

Factual Consistency Rate: The percentage of generated statements that are factually consistent with the retrieved documents.

Context Relevance Metrics

Context Precision: Measures what proportion of the retrieved context contains information relevant to answering the query.

Context Recall: Evaluates whether the retrieved context contains all the information needed to completely answer the query.

Context Relevance Score: Often assessed through human evaluation or specialized models that score the relevance of each retrieved passage to the query.

Human Evaluation Scales

Coherence: Rating the logical flow and readability of the generated response.

Relevance: Assessing how well the response addresses the original query.

Helpfulness: Measuring the practical utility of the response for the user's needs.

Preference Rating: Direct comparison between RAG system outputs and alternatives (like standard LLM responses).

Efficiency Metrics

Latency: End-to-end response time from query input to response generation.

Throughput: Number of queries that can be processed per unit of time.

Computational Cost: Resources required for operating the system (compute, memory, storage).

Integrated Evaluation Approaches

RAGAS: A framework specifically designed for RAG evaluation, combining metrics for faithfulness, answer relevance, and context relevance.

RAG Leaderboards: Standardized benchmarks comparing different RAG systems on common datasets.

A/B Testing: Comparing RAG system performance against baselines with real users.

Comprehensive evaluation of RAG systems typically involves a combination of these automatic metrics and human assessment to capture both technical performance and actual utility to end users.
        """,

        "vector_search.txt": """
Vector Similarity Search in RAG: Technical Deep Dive

Vector similarity search is a fundamental component of Retrieval-Augmented Generation (RAG) systems, enabling efficient identification of relevant content from large knowledge bases. Here's a comprehensive exploration of how this technology works:

Embedding Generation
The process begins with transforming text into numerical vector representations (embeddings):

Text Segmentation: Documents are divided into manageable chunks (paragraphs, passages, or semantic units).

Feature Extraction: Neural networks extract semantic features from text, capturing meaning beyond simple keywords.

Dimensionality Mapping: Text is mapped to high-dimensional vectors (typically 768 to 1536 dimensions), with each dimension representing a semantic feature.

Models: Common embedding models include BERT, Sentence Transformers, OpenAI's text-embedding models, or domain-specialized embedding models.

Consistency: Both documents and queries must use the same embedding model to ensure comparable vector spaces.

Similarity Metrics
Several mathematical approaches measure the "closeness" of vectors:

Cosine Similarity: Measures the cosine of the angle between vectors, ranging from -1 (opposite) to 1 (identical). Cosine similarity focuses on directional similarity rather than magnitude.

Euclidean Distance: Calculates the straight-line distance between two points in vector space. Smaller distances indicate greater similarity.

Dot Product: The sum of the products of corresponding vector components, often used with normalized vectors.

Manhattan Distance: The sum of the absolute differences between vector components, sometimes preferred for specific applications.

Efficient Retrieval Algorithms
Finding nearest neighbors in high-dimensional spaces presents computational challenges, addressed by:

Exact Nearest Neighbor (Exact NN): Guarantees finding the closest vectors but becomes computationally prohibitive for large collections.

Approximate Nearest Neighbor (ANN): Sacrifices perfect accuracy for dramatic speed improvements, usually finding most of the correct nearest neighbors.

Popular ANN Algorithms:
- Hierarchical Navigable Small World (HNSW): Creates a navigable graph structure for efficient traversal
- Inverted File Index (IVF): Divides the vector space into clusters for faster search
- Product Quantization (PQ): Compresses vectors to reduce memory requirements
- Locality-Sensitive Hashing (LSH): Maps similar items to the same "buckets" with high probability

Vector Database Implementation
Specialized databases optimize vector search operations:

Indexing Structures: Pre-computed data structures that accelerate similarity searches.

Clustering Methods: Grouping similar vectors to narrow search spaces.

Quantization Techniques: Compressing vector representations to reduce memory and computational requirements.

Sharding and Distribution: Partitioning vector data across multiple servers for horizontal scaling.

Metadata Filtering: Combining vector search with traditional filtering on categories, dates, or other attributes.

Retrieval Enhancements
Modern RAG systems employ several techniques to improve retrieval quality:

Hybrid Search: Combining vector similarity with keyword matching or BM25 for better results.

Query Expansion: Augmenting the original query with related terms to improve recall.

Dense Passage Retrieval: Specialized bi-encoder models that create separate embedding spaces for queries and documents.

Re-ranking: Two-stage retrieval where an initial broad search is refined by a more computationally intensive model.

Contextual Embeddings: Creating embeddings that account for the broader context of text chunks rather than isolated passages.

The effectiveness of vector similarity search in RAG largely determines the quality of the information provided to the language model, directly impacting the final generated response. Optimizing this component is critical for building high-performance RAG applications.
        """,

        "rag_challenges.txt": """
Challenges in Implementing Effective RAG Systems

Retrieval-Augmented Generation (RAG) systems face numerous technical and practical challenges that impact their effectiveness. This document explores the key obstacles and considerations when implementing production-grade RAG solutions.

Context Length Management
Challenge: Modern LLMs have token limits constraining how much retrieved information can be included.
Implications:
- Too little context leads to incomplete information
- Too much context wastes token capacity and increases costs
- Important information may be truncated
Solutions:
- Advanced chunking strategies that preserve semantic units
- Sophisticated ranking to prioritize the most relevant content
- Iterative approaches that process multiple context batches
- Summarization of retrieved documents before inclusion

Retrieval Quality Issues
Challenge: Ensuring retrieved documents contain relevant, accurate information for the query.
Factors:
- Vocabulary mismatch between queries and documents
- Handling ambiguous or broad questions
- Retrieving information that's factually accurate but outdated
Solutions:
- Hybrid retrieval combining semantic and lexical search
- Query reformulation and expansion
- Ensemble methods using multiple retrieval approaches
- Temporal awareness in document selection

Semantic Search Limitations
Challenge: Vector similarity doesn't always capture complex semantic relationships.
Problems:
- Difficulty with negations and contrasts
- Challenges with hypothetical or counterfactual questions
- Limited understanding of numerical reasoning requirements
Solutions:
- Multi-vector representations for documents
- Fine-tuning embedding models on domain-specific data
- Reranking with cross-encoder models that assess query-document pairs directly
- Incorporating structural and symbolic knowledge

Computational Efficiency
Challenge: Balancing performance with resource constraints.
Trade-offs:
- Index size vs. retrieval speed
- Embedding quality vs. computation time
- Response latency vs. throughput
Solutions:
- Vector compression techniques like Product Quantization
- Caching frequent queries and results
- Approximate nearest neighbor algorithms
- Tiered retrieval architectures with progressive refinement
- Hardware acceleration (GPUs, vector search accelerators)

Data Freshness and Management
Challenge: Maintaining up-to-date knowledge without constant reprocessing.
Difficulties:
- Determining when to update embeddings
- Tracking document versions and changes
- Handling conflicting information across documents
Solutions:
- Incremental indexing strategies
- Metadata-based freshness scoring
- Change detection to trigger selective updates
- Versioning systems for knowledge bases

Cross-lingual and Multimodal Challenges
Challenge: Handling diverse languages and content types.
Issues:
- Embedding quality varies across languages
- Multimodal content requires specialized processing
- Translation quality affects retrieval accuracy
Solutions:
- Multilingual embedding models
- Cross-lingual retrieval techniques
- Specialized processors for different content types (images, tables, code)
- Language-specific tuning of retrieval parameters

Hallucination Mitigation
Challenge: Preventing the LLM from generating facts not in the retrieved context.
Complications:
- Models blend retrieved information with parametric knowledge
- Difficult to verify all generated content against sources
- Trade-off between creativity and strict adherence to sources
Solutions:
- Explicit citations and source attribution
- Post-generation verification against retrieved content
- Focused fine-tuning for improved grounding
- Confidence scores for different parts of the response

Chunking Strategy Optimization
Challenge: Determining how to divide documents into retrievable pieces.
Considerations:
- Chunk size affects semantic coherence
- Overlap between chunks helps preserve context
- Different content types require different approaches
Solutions:
- Semantic chunking based on section boundaries
- Hierarchical chunking with multiple granularities
- Adaptive strategies based on content density
- Metadata-enriched chunks for improved retrieval

Evaluation Complexity
Challenge: Developing reliable metrics to assess RAG system quality.
Difficulties:
- Traditional IR metrics don't capture generation quality
- Generation metrics don't assess factual accuracy
- User satisfaction involves many subjective factors
Solutions:
- Composite evaluation frameworks (e.g., RAGAS)
- Factual consistency checking
- Source attribution accuracy assessment
- Human evaluation for qualitative aspects

Balancing Retrieval vs. Generation
Challenge: Finding the right mix of retrieved context and generative capability.
Trade-offs:
- Heavy reliance on retrieval may limit fluency
- Excessive generation freedom increases hallucination risk
- Domain-specific considerations affect optimal balance
Solutions:
- Adaptive prompting based on retrieval confidence
- Different strategies for different query types
- Explicit control parameters for retrieval influence
- Continuous learning from user feedback

Addressing these challenges requires a multifaceted approach combining algorithm innovation, system design optimization, and continuous evaluation. As RAG systems evolve, new solutions continue to emerge for these persistent challenges.
        """
    }

    # Save documents
    for filename, content in documents.items():
        file_path = os.path.join(output_folder, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())

    print(f"Saved {len(documents)} manually created documents to the {output_folder} directory")


def main():
    start_time = time.time()

    # User can choose whether to use manually created high-quality samples
    use_manual_samples = input("Use manually created high-quality samples for evaluation? (y/n): ").lower() == 'y' # Translated prompt

    if use_manual_samples:
        print("Using manually created high-quality samples...")
        create_manual_documents(output_folder="manual_documents")
        df = create_manual_test_samples(output_path="manual_rag_test.csv")

        print("\nYou can now use these manually created samples to evaluate the RAG system")
        print("Please use the following parameters during evaluation:")
        print("- Document directory: manual_documents")
        print("- Test questions file: manual_rag_test.csv")
    else:
        print("Using NQ dataset...")
        print("Starting download of Natural Questions dataset...")
        # Use a smaller dataset to speed up processing
        dataset = load_dataset("natural_questions", split="train[:3500]")

        print(f"Dataset loaded, took {time.time() - start_time:.2f} seconds")
        print(f"Dataset size: {len(dataset)} examples")

        # Preprocess dataset into question-answering format
        df = preprocess_nq_for_rag(dataset, sample_size=300)  # Reduce sample size to speed up processing

        # Prepare documents as knowledge base
        doc_map = prepare_nq_documents(dataset)

    print(f"All processing complete, total time taken {time.time() - start_time:.2f} seconds")

    # Display a few sample examples
    if len(df) > 0:
        print("\nExample Question-Answer Pairs:")
        for i in range(min(3, len(df))):
            print(f"\nQuestion {i + 1}: {df.iloc[i]['question']}")
            print(f"Answer: {df.iloc[i]['answer_1'][:200]}...") # Translated "答案"

    print("\nYou can now use this dataset to evaluate the RAG system")


if __name__ == "__main__":
    main()