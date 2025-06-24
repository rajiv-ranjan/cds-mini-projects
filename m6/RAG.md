# Learning on RAG

## Notes

- Embedding: Process of representing text (from docs etc) in dense vector form that also represents the semantic relationship between chunks. Helpful to store the chunks in the vector database. Also used to retrieve similar chunks from the vector db.
  - There are many pre-trained models available on HuggingFace that helps with embedding. https://huggingface.co/spaces/mteb/leaderboard
  - `TODO` Learn which model to use when?
  - In the below code; explore what options are there and there use
  ```python
    from langchain_huggingface import HuggingFaceEmbeddings
    
    modelPath ="mixedbread-ai/mxbai-embed-large-v1"                  # Model card: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
                                                                     # Find other Emb. models at: https://huggingface.co/spaces/mteb/leaderboard
    
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': device}      # cuda/cpu
    
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}
    
    embedding =  HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
  ```
- `TODO` While querying how can we combine both relevance and specificity?
  - MMR query:
  
  ```python
  vectordb.max_marginal_relevance_search(question, k=3, fetch_k=6)
  ```
  
  - Specific query:
  ```python
  vectordb.similarity_search(
    question,
    k=5,
    #filter={"source":'/content/pca_d1.pdf'} # manually passing metadata, using metadata filter.
    filter={"source":'/content/ens_d1.pdf'}
    )
  ```
  
- `TODO` BitsAndSytes can't work with macOS. We found that GGUF format and llama.cpp is superior way to run quantized model on macOS and other strong CPUs. Explore and write code to make it work. Read another md file [here](QuantizedModels.md).

- **The Modern "Rule of Thumb" for LangChain Imports**
The LangChain library was split into several packages to make it more modular. Here’s a simple way to remember where to import from:

  - **Core Abstractions** (`langchain-core`): For the fundamental building blocks of LangChain that don't require third-party libraries.

    - Imports from: `langchain_core`
    - Examples: Prompts (ChatPromptTemplate), Output Parsers (StrOutputParser), Runnables (RunnablePassthrough), Messages (HumanMessage).
  - **Dedicated Partner Integrations** (`langchain-<partner>`): For major partners that have their own dedicated package. This is the preferred way to use these integrations.
    - Imports from: `langchain_openai`, `langchain_ollama`, `langchain_huggingface`, etc.
    - Examples: ChatOpenAI, ChatOllama.
  - **Community Integrations** (`langchain-community`): For the vast collection of other third-party tools.

    - Imports from: `langchain_community`
    - Examples: Document Loaders (CSVLoader, PyPDFLoader), Vector Stores (FAISS, Chroma), other LLMs and tools.


## Key Packages

| **Package**               | **Notes** |
|---------------------------|-----------|
| `langchain`               | A framework for developing applications powered by LLMs, providing tools for chains, agents, memory, and retrieval-augmented generation (RAG). |
| `torch` (PyTorch)         | Deep learning library for training and deploying neural networks; foundational for many LLMs (e.g., Llama, GPT). |
| `transformers`            | Hugging Face library for state-of-the-art NLP models (BERT, GPT, etc.), including tokenization, training, and inference. |
| `sentencepiece`           | Tokenization library for subword units (used in models like Llama, T5); handles multilingual text efficiently. |
| `accelerate`              | Optimizes PyTorch code for multi-GPU/TPU training and inference with minimal code changes. |
| `bitsandbytes`            | Enables **8-bit/4-bit quantization** of models, reducing GPU memory usage (critical for running large LLMs). |
| `einops`                  | Simplifies tensor operations (reshaping, rearranging) in PyTorch/TensorFlow; used in transformer architectures. |
| `sentence-transformers`   | Framework for **embedding models** (e.g., BERT, MPNet) to generate semantic vector representations of text. |
| `langchain_community`     | Community-contributed integrations for LangChain (e.g., third-party tools, lesser-known LLMs). |
| `langchain_huggingface`   | Official LangChain-Hugging Face integration for Hugging Face models, pipelines, and endpoints. |
| `huggingface_hub`         | API/client for interacting with Hugging Face’s model hub (download/upload models, datasets). |
| `chromadb`                | Vector database for storing embeddings and enabling semantic search (used in RAG pipelines). |
| `langchain-huggingface`   | (Legacy) Older package for Hugging Face-LangChain integration; superseded by `langchain_huggingface`. |
| `hf_xet`                  | Experimental Hugging Face tool for efficient model/data caching (not widely adopted yet). |

## References

### Blogs

- [What is a RAG with examples?](https://www.superannotate.com/blog/rag-explained)
