import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from elasticsearch import Elasticsearch, helpers

class UniversalSentenceEncoder:
    def __init__(self):
        self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    def encode(self, texts):
        return self.embed(texts).numpy()

class vectorSearchClient:
    def __init__(self, docs, index_name):
        self.model = UniversalSentenceEncoder()
        self.es_client = Elasticsearch('http://localhost:9200/')
        self.index_name = index_name
        self.docs = docs
        self.docs = self.documents(docs)
        self.createIndexSettings()

    def documents(self, docs):
        documents = []

        for ind, doc in enumerate(docs):
            if doc["text"] is None:
                doc["text"] = "No text available"
            doc["text_vector"] = self.model.encode([doc["text"]]).tolist()[0]
            documents.append(doc)
            
        return documents

    def createIndexSettings(self):
        self.index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "section": {"type": "text"},
                    "question": {"type": "text"},
                    "course": {"type": "keyword"},
                    "text_vector": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"},
                }
            }
        }

        self.es_client.indices.delete(index=self.index_name, ignore_unavailable=True)
        self.es_client.indices.create(index=self.index_name, body=self.index_settings)

    # Generate actions for bulk indexing
    def generate_actions(self, local_docs):
        for i, doc in enumerate(local_docs):

            if not isinstance(doc, dict):
                print(f"Skipping document {i}: Not a dictionary")
                continue

            yield {
                "_index": self.index_name,
                "_id": i,  # Adding ID for tracking
                "_source": doc
            }

    def createBatchIndex(self):

        docs = self.docs

        # Index the documents with batches
        batch_size = 1000
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            success, failed = helpers.bulk(self.es_client, self.generate_actions(batch))
            print(f"Batch {i // batch_size + 1}: Indexed {success} documents")
            if failed:
                print("\nFailed documents:~~~~~~~~~~~~~~~")
                for error in failed:
                    print(f"Document ID: {error['index']['_id']}")
                    print(f"Error: {error['index']['error']}")

        return self.es_client

    def vector_search(self, query, top_k=5):
        vector_search_term = self.model.encode([query])
        query = {
            "field": "text_vector",
            "query_vector": vector_search_term,
            "k": top_k,
            "num_candidates": 10000,
        }

        res = self.es_client.search(index=self.index_name, knn=query, source=["text", "section", "question", "course"])
        results = res["hits"]["hits"]

        return results

    def vector_search_knn(self, query, top_k=5):
        vector_search_term = self.model.encode([query])
        knn_query = {
            "field": "text_vector",
            "query_vector": vector_search_term[0],
            "k": top_k,
            "num_candidates": 10000
        }

        response = self.es_client.search(
            index=self.index_name,
            query={
                "match": {"section": "General course-related questions"},
            },
            knn=knn_query,
            size=5
        )

        #results = response["hits"]["hits"]

        result_docs = []

        for hit in response['hits']['hits']:
            result_docs.append(hit['_source'])

        return result_docs