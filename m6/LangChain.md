# Learning

- I was getting 429 from huggingface as I tried opening the website.
  - **Use the Hugging Face Cache (Avoid Re-Downloading)**
Once a model is downloaded, Hugging Face libraries are smart enough to use a local cache on your computer so you don't have to download it again. The 429 error often happens when a script is misconfigured and tries to re-download on every run.
    - **How it works**: The first time you load a model (e.g., HuggingFaceEmbeddings(model_name="...")), the files are downloaded to a cache directory (usually ~/.cache/huggingface/hub on Mac/Linux). Subsequent calls will load the model directly from this local directory, which is nearly instant and makes no network requests.
    - **What to check**: Ensure your script isn't running in an environment where the cache is cleared on every run (like some misconfigured Docker containers).
  - I was blocked at the network level. Don't know why. 
    - **Solution** I moved to my companies VPN.
- `Open LLM Leaderboard` on HuggingFace.co: [here](https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-best-models-652d6c7965a4619fb5c27a03)
  - There scores are available here: [here](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)
- There are **different kinds of leaderboard**. All can be seen at this link: [here](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a) e.g. 
    - Open LLM Leaderboard
    - MTEB Leaderboard (Embedding)
    - Chatbot Arena Leaderboard
    - LLM-Perf Leaderboard
    - Big Code Models Leaderboard
    - ...
    - ...
- For `reasoning`; the first tier LLM like Claude Sonnet, Google Gemini etc performs a lot better vs open source model. This is crucial for Agent building.
- 