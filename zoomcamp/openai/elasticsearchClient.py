from elasticsearch import Elasticsearch, helpers

# build an elastic search client
class buildElasticSearchClient:
    def __init__(self, docs, index_name):
        self.es_client = Elasticsearch('http://localhost:9200/')
        self.index_name = index_name
        self.docs = docs # List of documents to index
        self.createIndexSettings()
        self.createIndices()

    # prepare the index settings
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
                    "course": {"type": "keyword"}
                }
            }
        }

    # create the inidices
    def createIndices(self):

        # Delete the index if it exists
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)

        # Create the index
        self.es_client.indices.create(index=self.index_name, body=self.index_settings)

    # Generate actions for bulk indexing
    def generate_actions(self):
        for doc in self.docs:
            yield {
                "_index": self.index_name,
                "_source": doc
            }

    #Create single index
    def createIndex(self):

        # Index the documents without batches
        self.es_client.index(index=self.index_name, document=self.docs)
        return self.es_client

    # Search the index
    def createBulkIndex(self):
        # Index the documents without batches
        success, failed = helpers.bulk(self.es_client, self.generate_actions())
        print(f"Successfully indexed {success} documents")

        return self.es_client

    def createBatchIndex(self):

        docs = self.docs

        # Index the documents with batches
        batch_size = 1000
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            success, failed = helpers.bulk(self.es_client, self.generate_actions())
            print(f"Batch {i // batch_size + 1}: Indexed {success} documents")

        return self.es_client