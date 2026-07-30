# Retriever service contract

Start a dense retriever with an E5-family embedding model, the paper corpus
revision, and a FAISS index. The HTTP service must expose `POST /retrieve`:

```json
{"queries": ["query"], "topk": 50, "return_scores": true}
```

The response is `{"result": [[{"document": {...}, "score": 0.0}]]}`.
The Actor receives only the first `visible_top_m` documents from each search.
Index, corpus, model, top-k, and visible top-m revisions belong in every run
manifest. No retrieval path or port is hard-coded by EviBack.