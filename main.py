# pip install faiss-cpu
# pip install pandas numpy matplotlib seaborn langchain_huggingface langchain_community faiss langchain_ollama transformers sentence_transformers tqdm
# pip install matplotlib
# pip install langchain_huggingface
# pip install langchain_ollama


import os
import time
import csv
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM
from tqdm import tqdm
from transformers import logging
from sentence_transformers import SentenceTransformer

# Set logging level to reduce noise
logging.set_verbosity_error()


class BasicRAGEvaluator:
    def __init__(self, model_name="gemma3:27b-it-q8_0", ollama_base_url="http://127.0.0.1:11434"):
        """Initialize the RAG evaluator"""
        print("Initializing RAG evaluator...")

        # Configure LLM
        self.model_name = model_name
        self.llm = OllamaLLM(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.1  # Low temperature for more deterministic answers
        )

        # Configure vector database and retriever
        self.vector_db = None
        self.retriever = None

        # Load sentence embedding model for similarity evaluation
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.has_sentence_model = True
            print("Sentence embedding model loaded, available for semantic similarity evaluation")
        except:
            self.has_sentence_model = False
            print("Warning: Failed to load sentence embedding model, semantic similarity evaluation will not be available")

        print(f"LLM model configured: {model_name}")

    def load_documents(self, docs_folder="nq_documents", chunk_size=500, chunk_overlap=50):
        """Load documents and create the vector database"""
        print(f"Loading documents from {docs_folder}...")

        # Check if directory exists
        if not os.path.exists(docs_folder):
            raise FileNotFoundError(f"Document directory {docs_folder} does not exist")

        # Load documents
        documents = []
        for filename in os.listdir(docs_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(docs_folder, filename)
                with open(file_path, "r", encoding='utf-8') as file:
                    text = file.read()
                    documents.append(Document(page_content=text, metadata={"source": filename}))

        print(f"Successfully loaded {len(documents)} documents")

        # Text chunking
        print("Starting text chunking process...")
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        docs = text_splitter.split_documents(documents)
        print(f"Text chunking complete, generated {len(docs)} text chunks, took {time.time() - start_time:.2f} seconds")

        # Create vector database
        print("Starting vector database creation...")
        start_time = time.time()
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        print("Embedding model loaded, starting document processing...")
        self.vector_db = FAISS.from_documents(docs, embeddings)
        print(f"Vector database creation complete, took {time.time() - start_time:.2f} seconds")

        return self

    def clean_retrieved_context(self, context):
        """Improved cleaning and formatting of retrieved context"""
        # Clean up extra spaces and newlines
        cleaned = re.sub(r'\s+', ' ', context).strip()

        # Fix common formatting issues
        cleaned = re.sub(r'(\d) - (\d)', r'\1-\2', cleaned)  # Fix date format
        cleaned = re.sub(r' ([.,;:?!])', r'\1', cleaned)  # Fix space before punctuation

        # Remove potential residual HTML entities
        cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)

        return cleaned

    def test_with_rag(self, question):
        """Answer the question using RAG enhancement"""
        if not self.vector_db:
            raise ValueError("Please load documents and create the vector database first")

        print(f"RAG enhancement test: '{question}'")
        start_time = time.time()

        # 1. Retrieve relevant documents
        docs = self.vector_db.similarity_search(question, k=3)
        raw_context = "\n\n".join([doc.page_content for doc in docs])

        # Clean and format the retrieved context
        context = self.clean_retrieved_context(raw_context)

        # 2. Build the prompt, including retrieved content
        prompt = f"""Please answer the question based on the following retrieved information. Use only the provided information and do not use your own knowledge.
If the retrieved information is insufficient to answer the question, please state directly "Based on the provided information, I cannot answer this question."

Retrieved Information:
{context}

Question: {question}

Answer:"""

        # 3. Use LLM to generate the answer
        response = self.llm.invoke(prompt)

        print(f"RAG enhanced answer generation complete, took {time.time() - start_time:.2f} seconds")

        # 4. Record the sources of retrieved documents for analysis
        doc_sources = [doc.metadata.get("source", "unknown") for doc in docs]

        return {
            "answer": response,
            "context": context,
            "raw_context": raw_context,
            "doc_sources": doc_sources,
            "time_taken": time.time() - start_time
        }

    def test_without_rag(self, question):
        """Answer the question using the standard LLM"""
        print(f"Standard LLM test: '{question}'")
        start_time = time.time()

        # Build the prompt
        prompt = f"Please answer the following question: {question}"

        # Use LLM to answer the question
        response = self.llm.invoke(prompt)

        print(f"Standard answer generation complete, took {time.time() - start_time:.2f} seconds")

        return {
            "answer": response,
            "time_taken": time.time() - start_time
        }

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate the semantic similarity between two texts"""
        if not self.has_sentence_model or not text1 or not text2:
            return 0.0

        try:
            # Calculate embeddings
            embedding1 = self.sentence_model.encode(text1)
            embedding2 = self.sentence_model.encode(text2)

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )

            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def calculate_lexical_overlap(self, text1, text2):
        """Calculate the lexical overlap rate between two texts"""
        if not text1 or not text2:
            return 0.0

        # Extract meaningful words (length > 3)
        words1 = set([word.lower() for word in re.findall(r'\b\w+\b', text1) if len(word) > 3])
        words2 = set([word.lower() for word in re.findall(r'\b\w+\b', text2) if len(word) > 3])

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0

    def evaluate_on_dataset(self, csv_path="nq_for_rag.csv", max_samples=5):
        """Evaluate the performance difference between RAG and standard LLM on a dataset"""
        print(f"Loading test questions from {csv_path}...")

        # Read CSV file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} does not exist")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} questions")

        # Limit the number of evaluation samples
        df = df.head(max_samples)

        # Prepare results storage
        results = []

        # Create directory to save detailed retrieval results
        retrieval_dir = "retrieval_results"
        if not os.path.exists(retrieval_dir):
            os.makedirs(retrieval_dir)

        # Evaluate each question
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
            question = row['question']
            reference_answer = row['answer_1']

            try:
                # Test standard LLM
                standard_result = self.test_without_rag(question)

                # Test RAG enhanced LLM
                rag_result = self.test_with_rag(question)

                # Calculate similarity and overlap evaluation metrics
                rag_ref_similarity = self.calculate_semantic_similarity(
                    rag_result["answer"], reference_answer
                )
                std_ref_similarity = self.calculate_semantic_similarity(
                    standard_result["answer"], reference_answer
                )

                rag_ref_overlap = self.calculate_lexical_overlap(
                    rag_result["answer"], reference_answer
                )
                std_ref_overlap = self.calculate_lexical_overlap(
                    standard_result["answer"], reference_answer
                )

                # Save result
                result_item = {
                    "question": question,
                    "reference_answer": reference_answer,
                    "standard_answer": standard_result["answer"],
                    "rag_answer": rag_result["answer"],
                    "retrieved_context": rag_result["context"],
                    "raw_context": rag_result["raw_context"],
                    "doc_sources": rag_result["doc_sources"],
                    "standard_time": standard_result["time_taken"],
                    "rag_time": rag_result["time_taken"],
                    "rag_reference_similarity": rag_ref_similarity,
                    "standard_reference_similarity": std_ref_similarity,
                    "similarity_improvement": rag_ref_similarity - std_ref_similarity,
                    "rag_reference_overlap": rag_ref_overlap,
                    "standard_reference_overlap": std_ref_overlap,
                    "overlap_improvement": rag_ref_overlap - std_ref_overlap
                }

                results.append(result_item)

                # Save intermediate results
                self.save_results(results, "evaluation_results.csv")

                # Save detailed retrieval results to a separate file
                retrieval_file = os.path.join(retrieval_dir, f"retrieval_{i + 1:02d}.txt")
                with open(retrieval_file, "w", encoding="utf-8") as f:
                    f.write(f"Question: {question}\n\n")
                    f.write(f"Reference Answer: {reference_answer}\n\n")
                    f.write(f"Retrieved Documents: {', '.join(rag_result['doc_sources'])}\n\n")
                    f.write("Raw Retrieved Content:\n")
                    f.write(rag_result["raw_context"])
                    f.write("\n\nProcessed Retrieved Content:\n")
                    f.write(rag_result["context"])
                    f.write("\n\nRAG Answer:\n")
                    f.write(rag_result["answer"])
                    f.write("\n\nStandard LLM Answer:\n")
                    f.write(standard_result["answer"])
                    f.write("\n\nSimilarity Evaluation:\n")
                    f.write(f"RAG vs Reference Similarity: {rag_ref_similarity:.4f}\n")
                    f.write(f"Standard LLM vs Reference Similarity: {std_ref_similarity:.4f}\n")
                    f.write(f"Similarity Improvement: {rag_ref_similarity - std_ref_similarity:.4f}\n")

                # Brief pause after each question to avoid API limits
                time.sleep(1)
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue

        print(f"Evaluation complete, evaluated {len(results)} questions")

        # Save complete evaluation data as JSON
        with open("evaluation_full_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved complete evaluation data to evaluation_full_results.json")

        return results

    def save_results(self, results, output_path="evaluation_results.csv"):
        """Save evaluation results to a CSV file"""
        with open(output_path, "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Write header row - added retrieved context and similarity evaluation metrics
            writer.writerow([
                "Question", "Reference Answer",
                "Standard Answer", "RAG Answer",
                "Retrieved Context",
                "Standard Time (s)", "RAG Time (s)",
                "RAG-Ref Similarity", "Std-Ref Similarity", "Similarity Improvement",
                "RAG-Ref Overlap", "Std-Ref Overlap", "Overlap Improvement"
            ])

            # Write result rows
            for r in results:
                ref_answer = r["reference_answer"]
                if len(ref_answer) > 150:
                    ref_answer = ref_answer[:150] + "..."

                # Truncate retrieved context, but keep more content
                context = r["retrieved_context"]
                if len(context) > 300:
                    context = context[:300] + "..."

                writer.writerow([
                    r["question"], ref_answer,
                    r["standard_answer"], r["rag_answer"],
                    context,
                    f"{r['standard_time']:.2f}", f"{r['rag_time']:.2f}",
                    f"{r.get('rag_reference_similarity', 0):.4f}",
                    f"{r.get('standard_reference_similarity', 0):.4f}",
                    f"{r.get('similarity_improvement', 0):.4f}",
                    f"{r.get('rag_reference_overlap', 0):.4f}",
                    f"{r.get('standard_reference_overlap', 0):.4f}",
                    f"{r.get('overlap_improvement', 0):.4f}"
                ])

        print(f"Saved evaluation results to {output_path}")

    def visualize_results(self, results):
        """Visualize evaluation results"""
        if not results:
            print("No results to visualize")
            return

        # Create visualization directory
        viz_dir = "evaluation_visualizations"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # Calculate average response time
        standard_times = [r["standard_time"] for r in results]
        rag_times = [r["rag_time"] for r in results]

        avg_standard_time = np.mean(standard_times)
        avg_rag_time = np.mean(rag_times)

        # 1. Response time comparison chart
        plt.figure(figsize=(10, 6))
        plt.bar(['Standard LLM', 'RAG Enhanced'], [avg_standard_time, avg_rag_time])
        plt.title(f'Average Response Time ({self.model_name})')
        plt.ylabel('Time (seconds)')
        plt.savefig(os.path.join(viz_dir, 'response_time_comparison.png'))
        plt.close()

        # 2. Response length comparison
        standard_lengths = [len(r["standard_answer"]) for r in results]
        rag_lengths = [len(r["rag_answer"]) for r in results]

        avg_standard_length = np.mean(standard_lengths)
        avg_rag_length = np.mean(rag_lengths)

        plt.figure(figsize=(10, 6))
        plt.bar(['Standard LLM', 'RAG Enhanced'], [avg_standard_length, avg_rag_length])
        plt.title(f'Average Response Length ({self.model_name})')
        plt.ylabel('Characters')
        plt.savefig(os.path.join(viz_dir, 'response_length_comparison.png'))
        plt.close()

        # 3. Similarity evaluation comparison
        if self.has_sentence_model:
            rag_similarities = [r.get('rag_reference_similarity', 0) for r in results]
            std_similarities = [r.get('standard_reference_similarity', 0) for r in results]
            improvements = [r.get('similarity_improvement', 0) for r in results]

            avg_rag_similarity = np.mean(rag_similarities)
            avg_std_similarity = np.mean(std_similarities)
            avg_improvement = np.mean(improvements)

            # 3.1 Average similarity comparison
            plt.figure(figsize=(10, 6))
            plt.bar(['Standard LLM', 'RAG Enhanced'], [avg_std_similarity, avg_rag_similarity])
            plt.title('Average Semantic Similarity with Reference')
            plt.ylabel('Similarity Score')
            plt.savefig(os.path.join(viz_dir, 'similarity_comparison.png'))
            plt.close()

            # 3.2 Similarity improvement distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(improvements, kde=True)
            plt.axvline(0, color='r', linestyle='--')
            plt.title('Distribution of Similarity Improvements')
            plt.xlabel('Improvement (RAG - Standard)')
            plt.savefig(os.path.join(viz_dir, 'similarity_improvement_distribution.png'))
            plt.close()

            # 3.3 Scatter plot: Per-question similarity comparison
            plt.figure(figsize=(10, 6))
            plt.scatter(std_similarities, rag_similarities, alpha=0.7)

            # Add diagonal line, representing no improvement
            max_value = max(max(std_similarities), max(rag_similarities))
            min_value = min(min(std_similarities), min(rag_similarities))
            plt.plot([min_value, max_value], [min_value, max_value], 'r--')

            plt.xlabel('Standard LLM Similarity')
            plt.ylabel('RAG Similarity')
            plt.title('Per-Question Similarity Comparison')
            plt.savefig(os.path.join(viz_dir, 'per_question_similarity.png'))
            plt.close()

        # 4. Lexical overlap evaluation comparison
        rag_overlaps = [r.get('rag_reference_overlap', 0) for r in results]
        std_overlaps = [r.get('standard_reference_overlap', 0) for r in results]
        overlap_improvements = [r.get('overlap_improvement', 0) for r in results]

        avg_rag_overlap = np.mean(rag_overlaps)
        avg_std_overlap = np.mean(std_overlaps)
        avg_overlap_improvement = np.mean(overlap_improvements)

        plt.figure(figsize=(10, 6))
        plt.bar(['Standard LLM', 'RAG Enhanced'], [avg_std_overlap, avg_rag_overlap])
        plt.title('Average Lexical Overlap with Reference')
        plt.ylabel('Overlap Score')
        plt.savefig(os.path.join(viz_dir, 'overlap_comparison.png'))
        plt.close()

        # 5. Overall evaluation summary table
        summary_data = {
            'Metric': ['Response Time (s)', 'Response Length', 'Semantic Similarity', 'Lexical Overlap'],
            'Standard LLM': [avg_standard_time, avg_standard_length, avg_std_similarity, avg_std_overlap],
            'RAG Enhanced': [avg_rag_time, avg_rag_length, avg_rag_similarity, avg_rag_overlap],
            'Difference': [
                avg_rag_time - avg_standard_time,
                avg_rag_length - avg_standard_length,
                avg_improvement,
                avg_overlap_improvement
            ],
            'Improvement %': [
                'N/A',
                f"{(avg_rag_length - avg_standard_length) / avg_standard_length * 100:.2f}%",
                f"{avg_improvement / max(0.0001, avg_std_similarity) * 100:.2f}%",
                f"{avg_overlap_improvement / max(0.0001, avg_std_overlap) * 100:.2f}%"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(viz_dir, 'evaluation_summary.csv'), index=False)

        # Create HTML formatted evaluation report
        html_summary = "<html><head><style>table {border-collapse: collapse; width: 100%;} th, td {text-align: left; padding: 8px; border: 1px solid #ddd;} tr:nth-child(even) {background-color: #f2f2f2;} th {background-color: #4CAF50; color: white;}</style></head><body>"
        html_summary += "<h1>RAG System Evaluation Summary</h1>"
        html_summary += f"<p>Evaluation Date: {time.strftime('%Y-%m-%d')}</p>"
        html_summary += f"<p>LLM Model: {self.model_name}</p>"
        html_summary += f"<p>Number of Questions Evaluated: {len(results)}</p>"

        # Add summary table
        html_summary += "<h2>Performance Metrics Summary</h2>"
        html_summary += "<table>"
        html_summary += "<tr><th>Metric</th><th>Standard LLM</th><th>RAG System</th><th>Difference</th><th>Improvement Percentage</th></tr>"

        for i in range(len(summary_df)):
            row = summary_df.iloc[i]
            html_summary += f"<tr><td>{row['Metric']}</td><td>{row['Standard LLM']:.4f}</td><td>{row['RAG Enhanced']:.4f}</td><td>{row['Difference']:.4f}</td><td>{row['Improvement %']}</td></tr>"

        html_summary += "</table>"

        # Add question samples
        html_summary += "<h2>Example Question-Answer Comparison</h2>"
        for i, r in enumerate(results[:3]):  # Show only the first 3
            html_summary += f"<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #ddd;'>"
            html_summary += f"<p><strong>Question {i + 1}:</strong> {r['question']}</p>"
            html_summary += f"<p><strong>Reference Answer:</strong> {r['reference_answer'][:200]}...</p>"
            html_summary += f"<p><strong>Retrieved Context:</strong> {r['retrieved_context'][:200]}...</p>"
            html_summary += f"<p><strong>RAG Answer:</strong> {r['rag_answer'][:200]}...</p>"
            html_summary += f"<p><strong>Standard Answer:</strong> {r['standard_answer'][:200]}...</p>"
            html_summary += f"<p><strong>Semantic Similarity:</strong> RAG={r.get('rag_reference_similarity', 0):.4f}, Standard={r.get('standard_reference_similarity', 0):.4f}, Improvement={r.get('similarity_improvement', 0):.4f}</p>"
            html_summary += "</div>"

        html_summary += "<p>Please check the CSV file and retrieval results directory for full evaluation results.</p>"
        html_summary += "</body></html>"

        with open(os.path.join(viz_dir, 'evaluation_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_summary)

        print("Results visualization saved to the " + viz_dir + " directory")

        # Print sample comparison
        print("\n=== Question-Answer Sample Comparison ===")
        for i, r in enumerate(results[:3]):  # Show only the first 3 results
            print(f"\nQuestion {i + 1}: {r['question']}")
            print(f"Standard Answer: {r['standard_answer'][:200]}...")
            print(f"RAG Answer: {r['rag_answer'][:200]}...")
            if self.has_sentence_model:
                print(
                    f"Semantic Similarity: RAG={r.get('rag_reference_similarity', 0):.4f}, Standard={r.get('standard_reference_similarity', 0):.4f}")
            print("-" * 50)


def main():
    print("RAG evaluation program starting execution...")

    # Check if Ollama is available
    ollama_url = "http://127.0.0.1:11434"
    print(f"Please ensure Ollama is running at {ollama_url}")

    try:
        # Initialize the evaluator
        evaluator = BasicRAGEvaluator(model_name="gemma3:12b-it-fp16", ollama_base_url=ollama_url)

        # Load documents
        evaluator.load_documents(docs_folder="nq_documents", chunk_size=500, chunk_overlap=50)

        # Evaluate on the dataset - reduce sample size for faster testing
        results = evaluator.evaluate_on_dataset(csv_path="nq_for_rag.csv", max_samples=300)

        # Visualize results
        evaluator.visualize_results(results)

        print("Evaluation program execution complete!")
        print("You can view the following content:")
        print("1. evaluation_results.csv - Contains simplified evaluation results")
        print("2. evaluation_full_results.json - Contains complete evaluation details")
        print("3. retrieval_results/ - Contains detailed retrieval content for each question")
        print("4. evaluation_visualizations/ - Contains evaluation results visualization and HTML report")

    except Exception as e:
        print(f"Evaluation program execution error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()