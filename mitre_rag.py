

import json
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import os


def download_mitre_attack_data():
    """Download MITRE ATT&CK Enterprise data"""
    print("Downloading MITRE ATT&CK data...")
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    response = requests.get(url)
    return response.json()

def extract_techniques(attack_data: dict) -> List[Dict]:
    """Extract techniques, tactics, and threat actors from ATT&CK data"""
    documents = []
    
    for obj in attack_data['objects']:
        if obj['type'] == 'attack-pattern':
            doc = {
                'id': obj.get('external_references', [{}])[0].get('external_id', 'N/A'),
                'name': obj.get('name', ''),
                'description': obj.get('description', ''),
                'tactics': [phase['phase_name'] for phase in obj.get('kill_chain_phases', [])],
                'type': 'technique'
            }
            documents.append(doc)
        
        elif obj['type'] == 'intrusion-set':
            doc = {
                'id': obj.get('external_references', [{}])[0].get('external_id', 'N/A'),
                'name': obj.get('name', ''),
                'description': obj.get('description', ''),
                'type': 'threat-actor'
            }
            documents.append(doc)
        
        elif obj['type'] == 'malware' or obj['type'] == 'tool':
            doc = {
                'id': obj.get('external_references', [{}])[0].get('external_id', 'N/A'),
                'name': obj.get('name', ''),
                'description': obj.get('description', ''),
                'type': obj['type']
            }
            documents.append(doc)
    
    return documents


def create_vector_database(documents: List[Dict], db_path="./mitre_attack_db"):
    """Create and populate ChromaDB with MITRE ATT&CK data"""
    
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Initializing ChromaDB...")
    
    client = chromadb.PersistentClient(path=db_path)
    
    
    try:
        client.delete_collection(name="mitre_attack")
    except:
        pass
    
   
    collection = client.create_collection(
        name="mitre_attack",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Creating embeddings for {len(documents)} documents...")
    print("This may take a few minutes...")
    
    # Process in batches for better performance
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        texts = []
        metadatas = []
        ids = []
        
        for j, doc in enumerate(batch):
            # Create text representation
            text = f"{doc['name']}: {doc['description']}"
            metadata = {
                'id': doc['id'],
                'name': doc['name'],
                'type': doc['type']
            }
            
            texts.append(text)
            metadatas.append(metadata)
            ids.append(f"doc_{i+j}")
        
       
        embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()
        
       
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
    
    print("✓ Vector database created successfully!")
    return collection, embedding_model



class MITREAttackRAG:
    """RAG system for querying MITRE ATT&CK knowledge base"""
    
    def __init__(self, collection, embedding_model, llm_model="llama3.2:3b"):
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
       
        print(f"\nChecking if {llm_model} is available...")
        try:
            ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': 'test'}])
            print(f"✓ {llm_model} is ready!")
        except Exception as e:
            print(f"✗ Error: {llm_model} not found!")
            print(f"\nPlease run: ollama pull {llm_model}")
            raise e
    
    def retrieve(self, query: str, n_results: int = 5):
        """Retrieve relevant documents from vector database"""
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results
    
    def generate_answer(self, query: str, context: str):
        """Generate answer using local LLM"""
        prompt = f"""You are a cybersecurity expert assistant with deep knowledge of the MITRE ATT&CK framework.

Context from MITRE ATT&CK:
{context}

Question: {query}

Provide a detailed, accurate answer based on the context above. If the context doesn't contain enough information, say so clearly. Include relevant technique IDs (like T1234) when applicable. Be concise but comprehensive."""

        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def query(self, question: str, n_results: int = 5):
        """Complete RAG pipeline: retrieve + generate"""
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)
        
        
        print("→ Retrieving relevant information from MITRE ATT&CK...")
        results = self.retrieve(question, n_results=n_results)
        
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context_parts.append(f"[{meta['id']}] {meta['name']}\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        
        print("→ Generating answer with Llama 3.2...")
        answer = self.generate_answer(question, context)
        
        print(f"\nAnswer:\n{answer}")
        
        print(f"\nSources:")
        for source in results['metadatas'][0]:
            print(f"  • [{source['id']}] {source['name']} ({source['type']})")
        
        return {
            'answer': answer,
            'sources': results['metadatas'][0],
            'context': context
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the RAG agent"""
    
    print("="*70)
    print("MITRE ATT&CK RAG Agent")
    print("="*70)
    
    db_path = "./mitre_attack_db"
    
    if os.path.exists(db_path) and os.path.exists(f"{db_path}/chroma.sqlite3"):
        print("\n✓ Found existing vector database!")
        print("Loading database...")
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="mitre_attack")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    else:
        print("\n→ No existing database found. Creating new one...")
        
    
        attack_data = download_mitre_attack_data()
        documents = extract_techniques(attack_data)
        print(f"✓ Extracted {len(documents)} documents from MITRE ATT&CK")
        
        
        collection, embedding_model = create_vector_database(documents, db_path)
    
    
    print("\n" + "="*70)
    print("Initializing RAG System...")
    print("="*70)
    rag = MITREAttackRAG(collection, embedding_model, llm_model="llama3.2:3b")
    
    # Step 4: Example queries
    print("\n" + "="*70)
    print("Running Example Queries")
    print("="*70)
    
    example_questions = [
        "What is credential dumping and how do attackers use it?",
        "What techniques are used for lateral movement?",
        "What is T1059 and what are its sub-techniques?",
    ]
    
    for question in example_questions:
        rag.query(question)
        print("\n")
    
  
    print("\n" + "="*70)
    print("Interactive Mode - Ask your own questions!")
    print("Type 'quit' or 'exit' to stop")
    print("="*70)
    
    while True:
        user_question = input("\nYour question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_question:
            continue
        
        try:
            rag.query(user_question)
        except KeyboardInterrupt:
            print("\n\nInterrupted! Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
