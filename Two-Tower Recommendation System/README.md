🧠 Two-Tower Recommendation System using FAISS
📌 Project Overview

This project implements a scalable product recommendation system using a Two-Tower Neural Network architecture. It converts both user queries and products into dense vector embeddings and performs fast similarity search using FAISS.

🚀 Key Features
Two-Tower Neural Network (Query + Product)
Embedding compression (768 → 16 dimensions)
L2 normalization for cosine similarity
FAISS-based fast nearest neighbor search
Scalable retrieval for large product catalogs
Top-K product recommendations
🏗️ System Architecture
User Query
    ↓
Query Tower (32 → 16 embedding)
    ↓
FAISS Similarity Search
    ↓
Product Embeddings (precomputed)
    ↓
Top-K Product Results
🧠 Model Details
Query Tower:
Input: 32-d vector
Output: 16-d embedding
Product Tower:
Input: 160-d vector
Output: 16-d embedding
Similarity:
Cosine similarity using dot product on normalized vectors
⚙️ Tech Stack
Python
TensorFlow / Keras
FAISS (Facebook AI Similarity Search)
NumPy
Pandas
📊 Workflow
1. Data Preparation
Product metadata + embeddings
Query-product interaction dataset
2. Model Training
Binary classification (relevant / not relevant)
Loss: Binary Crossentropy
3. Embedding Generation
Product embeddings precomputed offline
Query embeddings computed at runtime
4. Indexing
FAISS Index built on product embeddings
5. Retrieval
Query embedding → FAISS search → Top-K products
⚡ How to Run
pip install faiss-cpu tensorflow pandas numpy
# Load FAISS index
index = faiss.read_index("product_model.faiss")

# Encode query
query_vec = query_model.predict(query_input)

# Normalize
faiss.normalize_L2(query_vec)

# Search
D, I = index.search(query_vec, k=10)
📦 Output

Returns:

Top 10 most relevant products
Product titles
Similarity scores
📈 Future Improvements
Reranking model (Deep Learning based)
Hybrid search (BM25 + FAISS)
Real-time user feedback learning
API deployment using FastAPI
Frontend using Streamlit
💡 Real-World Use Case

Used in:

E-commerce search (Amazon, Flipkart style)
Recommendation systems
Semantic product search
Large-scale retrieval systems
🏁 Conclusion

This project demonstrates a full end-to-end retrieval system using deep learning + approximate nearest neighbor search, capable of scaling to large product catalogs efficiently.
