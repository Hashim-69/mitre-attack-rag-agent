🛡️ MITRE ATT&CK RAG Agent

An AI-powered cybersecurity intelligence system that enables natural language querying of the MITRE ATT&CK framework using Retrieval-Augmented Generation (RAG)

This project was developed and uploaded using the Model Context Protocol (MCP) - showcasing modern AI-assisted development workflows.

🎯 Overview
This RAG (Retrieval-Augmented Generation) agent transforms how security professionals interact with MITRE ATT&CK knowledge. Instead of manually browsing through techniques and tactics, ask questions in natural language and get instant, contextually-aware answers with source attribution.
Why This Matters

Speed: Query 1,700+ techniques in seconds vs. manual website navigation
Intelligence: Semantic search understands intent, not just keywords
Accuracy: Answers grounded in official MITRE ATT&CK data with source citations
Accessibility: Natural language interface for both experts and learners
Privacy: 100% local deployment - no data leaves your machine


✨ Key Features
🧠 Advanced RAG Architecture

Vector Database: ChromaDB with cosine similarity search
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
LLM: Llama 3.2 3B quantized for CPU inference
Data Source: Official MITRE ATT&CK STIX/JSON feeds

🚀 Performance Optimized

Runs on consumer hardware (16GB RAM, no GPU required)
Sub-second retrieval from 1,700+ documents
5-15 second end-to-end query latency
Batch processing with memory-efficient embeddings

🔒 Security-First Design

Local inference - no external API calls
Persistent vector storage for offline operation
Automated data pipeline from MITRE's official repository


🎬 Demo
bash$ python mitre_rag.py

Question: What is credential dumping and how do attackers use it?
→ Retrieving relevant information from MITRE ATT&CK...
→ Generating answer with Llama 3.2...

Answer:
Credential dumping (T1003) is a technique where attackers extract account 
credentials from operating systems and software. Attackers commonly target 
LSASS memory (T1003.001), SAM database (T1003.002), and NTDS.dit from domain 
controllers (T1003.003). These credentials enable lateral movement and 
persistence in the network...

Sources:
  • [T1003] OS Credential Dumping (technique)
  • [T1003.001] LSASS Memory (technique)
  • [T1003.002] Security Account Manager (technique)

🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│                    User Query (Natural Language)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Embedding Model (all-MiniLM-L6-v2)                         │
│  Converts query to 384-dimensional vector                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  ChromaDB Vector Database                                    │
│  Semantic search across 1,700+ MITRE ATT&CK documents       │
│  Returns top-K most relevant techniques/tactics              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Context Assembly                                            │
│  Formats retrieved documents with metadata                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Llama 3.2 3B (Local LLM via Ollama)                        │
│  Generates contextually-aware answer                         │
│  Includes technique IDs and citations                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Final Answer + Source Attribution                           │
└─────────────────────────────────────────────────────────────┘

🚀 Quick Start
Prerequisites

Python 3.8+
16GB RAM (recommended)
10GB free disk space
Internet connection (first run only)

Installation
1. Clone the repository
bashgit clone https://github.com/Hashim-69/mitre-attack-rag-agent.git
cd mitre-attack-rag-agent
2. Install Python dependencies
bashpip install sentence-transformers chromadb ollama requests
3. Install Ollama and download the model
Windows/Mac/Linux:

Visit ollama.ai and install
Download the model:

bashollama pull llama3.2:3b
4. Run the agent
bashpython mitre_rag.py
First Run
On first execution, the system will:

Download MITRE ATT&CK data (~15MB)
Generate embeddings for 1,700+ documents (2-3 minutes)
Create persistent vector database
Run example queries
Enter interactive mode

Subsequent runs load instantly from the cached database!

💡 Usage Examples
Example Queries
python# Technique-specific questions
"What is T1059 and what are its sub-techniques?"
"How does PowerShell execution work in attacks?"

# Tactic-based questions
"What techniques are used for lateral movement?"
"Show me privilege escalation methods"

# Threat intelligence
"What tools does APT29 use?"
"How do ransomware groups typically operate?"

# Detection and mitigation
"How can I detect credential dumping?"
"What are the best defenses against phishing?"
Interactive Mode
After example queries, the agent enters interactive mode:
bashYour question: What is Pass-the-Hash?
→ Retrieving relevant information...
→ Generating answer...

[Detailed answer with technique IDs and sources]

Your question: quit
Goodbye!

🛠️ Technical Details
Technology Stack
ComponentTechnologyPurposeLLMLlama 3.2 3B (Quantized)Answer generationEmbeddingsall-MiniLM-L6-v2Text vectorizationVector DBChromaDBSemantic searchData SourceMITRE ATT&CK STIX JSONKnowledge baseLLM RuntimeOllamaLocal inference engine
Performance Metrics

Database Size: ~500MB (embeddings + metadata)
Embedding Generation: 2-3 minutes (1,700+ docs)
Query Latency: 5-15 seconds end-to-end
Retrieval Time: <1 second
Memory Usage: 4-6GB during inference
Documents Indexed: 1,700+ (techniques, tactics, groups, tools)

Model Selection Rationale
Llama 3.2 3B was chosen for:

Efficient CPU inference on consumer hardware
Strong instruction-following capabilities
Adequate security domain knowledge
2-3GB memory footprint (quantized)

all-MiniLM-L6-v2 was chosen for:

Fast CPU embedding generation
Good semantic understanding
Small model size (~80MB)
Proven performance on similarity tasks


📊 Data Pipeline
MITRE ATT&CK Data Ingestion
python# Automated extraction of:
- 600+ Attack Techniques
- 200+ Sub-Techniques  
- 140+ Threat Groups
- 700+ Software (malware/tools)
- 14 Tactics (kill chain phases)
Vectorization Strategy

Chunking: Technique name + full description
Batch Size: 100 documents per batch
Embedding Dimension: 384
Similarity Metric: Cosine similarity
Top-K Retrieval: 5 most relevant documents


🔧 Configuration
Customization Options
Change LLM Model:
pythonrag = MITREAttackRAG(collection, embedding_model, llm_model="mistral:7b")
Adjust Retrieval Count:
pythonresult = rag.query(question, n_results=10)  # Retrieve top 10 docs
Change Database Path:
pythoncollection, model = create_vector_database(documents, db_path="./custom_db")

🎓 Educational Value
This project demonstrates:
✅ RAG Architecture - Production-grade retrieval-augmented generation
✅ Vector Databases - Semantic search with embeddings
✅ Local LLM Deployment - Privacy-preserving AI inference
✅ Security Domain - Applied AI in cybersecurity
✅ Data Engineering - ETL pipeline for structured intelligence
✅ Optimization - Resource-constrained model deployment

🚧 Future Enhancements

 MITRE D3FEND Integration - Add defensive countermeasures mapping
 Multi-turn Conversations - Contextual follow-up questions
 Web Interface - Streamlit/Gradio UI for accessibility
 Export Functionality - Generate reports in PDF/JSON
 Real-time Updates - Automated MITRE ATT&CK synchronization
 Advanced Filtering - By tactic, platform, or data source
 Hybrid Search - Combine semantic + keyword search
 Fine-tuned Model - Domain-specific LLM training


🤝 Contributing
Contributions are welcome! Areas of interest:

Additional data sources (CAPEC, CVE, etc.)
Performance optimizations
UI/UX improvements
Documentation enhancements
Test coverage

Please open an issue before starting major work.

📝 License
MIT License - see LICENSE file for details

🙏 Acknowledgments

MITRE Corporation - For the ATT&CK framework and open data
Ollama - For making local LLM inference accessible
Sentence Transformers - For efficient embedding models
ChromaDB - For the vector database infrastructure
Model Context Protocol (MCP) - Development methodology used


📞 Contact
Created by Hashim
Note: This project was developed using Claude AI with the Model Context Protocol (MCP), demonstrating the power of AI-assisted software development. The MCP enabled seamless integration of code generation, debugging, and Git operations within a single conversational interface.

⭐ Star This Repo
If you find this project useful for learning or work, please give it a star! It helps others discover the project.

Built with 🤖 AI-Assisted Development | Powered by 🦙 Llama 3.2 | Secured by 🛡️ MITRE ATT&CK
