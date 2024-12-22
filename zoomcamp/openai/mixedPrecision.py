import tensorflow as tf
import os
import gc
import json

import minisearch
from elasticsearchClient import buildElasticSearchClient
#from elasticsearch import Elasticsearch, helpers

# FlanT5
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from transformers import AutoConfig, TFAutoModelForSeq2SeqLM, T5Tokenizer

# Mistral
from transformers import AutoTokenizer, TFAutoModelForCausalLM

from tensorflow.keras import mixed_precision

# Rag model class
class ragModel:

    def __init__(self, file, query, elastic=False):
        self.file = file
        self.query = query
        self.elastic = elastic
        docs = self.readjson()
        self.index_name = "course-questions"

        if self.elastic:

            self.es_client = buildElasticSearchClient(docs, self.index_name).createBatchIndex()
            #self.es_client = buildElasticSearchClient(docs, self.index_name).createBulkIndex()

        else:

            index = minisearch.Index(
                text_fields=["question", "text", "section"],
                keyword_fields=["course"]
            )
            index.fit(docs)
            self.index = index

    def readjson(self):

        file = self.file
        with open(file, 'rt') as f:
            data_raw = json.load(f)

        documents = []

        for course_dict in data_raw:
            for ind, doc in enumerate(course_dict['documents']):
                doc['course'] = course_dict['course']
                documents.append(doc)

        return documents

    def elastic_search(self):
        query = self.query
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": "data-engineering-zoomcamp"
                        }
                    }
                }
            }
        }

        response = self.es_client.search(index=self.index_name, body=search_query)

        result_docs = []

        for hit in response['hits']['hits']:
            result_docs.append(hit['_source'])

        return result_docs

    def search(self):
        if self.elastic:
            print("Using Elastic Search")
            results = self.elastic_search()
        else:
            boost = {'question': 3.0, 'section': 0.5}
            index = self.index
            query = self.query
            results = index.search(
                query=query,
                filter_dict={'course': 'data-engineering-zoomcamp'},
                boost_dict=boost,
                num_results=5
            )

        return results

    def build_prompt(self):
        query = self.query
        search_results = self.search()
        print('***************'); print(search_results)
        prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT:
        {context}
        """.strip()

        context = ""

        for doc in search_results:
            context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

        prompt = prompt_template.format(question=query, context=context).strip()
        return prompt

    def getModel(self):

        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')  # or 'mixed_bfloat16'

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        model = TFT5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
        return tokenizer, model

    def getQuantizedModel(self):
        from transformers import AutoConfig, TFAutoModelForSeq2SeqLM

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        # Load in bfloat16
        config = AutoConfig.from_pretrained("google/flan-t5-xl")
        model = TFAutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-xl",
            config=config,
            dtype=tf.bfloat16
        )
        return tokenizer, model

    def getsmallerModel(self):
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = TFT5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        return tokenizer, model

    def getmp(self):
        #from mixedPrecision import mixedPrecisionModel
        mp = mixedPrecisionModel()
        tokenizer, model = mp.initialize_model()
        return tokenizer, model

    def llm(self, generate_params=None):
        prompt = self.build_prompt()

        if generate_params is None:
            generate_params = {}

        # Get the pretrained tokenozer and model
        # tokenizer, model = getModel()
        # tokenizer, model = getQuantizedModel()
        # tokenizer, model = getsmallerModel()
        # tokenizer, model = getmp()
        tokenizer, model = self.getmp()

        input_ids = tokenizer(prompt, return_tensors="tf").input_ids  # .to("cuda")
        outputs = model.generate(
            input_ids,
            max_length=100,
            num_beams=5,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def chunkLlm(self):
        #from mixedPrecision import chunkPrompt

        tokenizer, model = self.getQuantizedModel()
        # tokenizer, model = self.getmp()
        #tokenizer, model = self.getModel()
        cmp = chunkPrompt(tokenizer, model)

        query  = self.query

        prompt = self.build_prompt()

        answer = cmp.rag(query, prompt)

        return answer

    def rag(self):

        # Generate the answer by using the prompt to feed the llm
        # answer = self.llm()

        answer = self.chunkLlm()

        return answer

# If prompt is too large, chunk and tokenize and combine the responses
class chunkPrompt:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model


    def chunk_text(self, text, max_chunk_size=450):  # Leave room for query

        tokenizer = self.tokenizer

        # Tokenize the full text
        tokens = tokenizer(text, return_tensors="tf", truncation=False)

        # If text fits in one chunk, return as is
        if len(tokens.input_ids[0]) <= max_chunk_size:
            return [text]

        # Split into sentences (simple split, you might want more sophisticated splitting)
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Check length of this sentence
            sentence_tokens = tokenizer(sentence, return_tensors="tf")
            sentence_length = len(sentence_tokens.input_ids[0])

            if current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def generate_text(self, prompt, max_input_length=512, max_output_length=128):
        # Tokenize with truncation
        tokenizer = self.tokenizer
        inputs = tokenizer(
            prompt,
            max_length=max_input_length,
            truncation=True,  # Explicitly enable truncation
            padding='max_length',  # Add padding
            return_tensors="tf"
        )

        model = self.model
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_output_length,
            num_beams=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def rag(self, query, context, max_input_length=512):
        # Chunk the context if needed
        context_chunks = self.chunk_text(context)
        responses = []

        for chunk in context_chunks:
            # Combine query with context chunk
            prompt = f"Context: {chunk}\nQuery: {query}"

            # Generate response for this chunk
            response = self.generate_text(
                prompt,
                max_input_length=max_input_length
            )
            responses.append(response)

        # Combine responses (you might want to customize this)
        return " ".join(responses)


# Mixed precision model to load progressively
class mixedPrecisionModel:

    def __init__(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Optionally, limit GPU memory
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpus[0],
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]  # 14GB
                # )
            except RuntimeError as e:
                print(f"GPU setup error: {e}")

        # 2. Enable TF32 on Ampere GPUs (like RTX 4080)
        #tf.config.experimental.enable_tensor_float_32_execution(True)

        # 3. Set up mixed precision
        #mixed_precision.set_global_policy('mixed_float16')  # or 'mixed_bfloat16'

        # 4. Configure minimal T5 settings
        self.config = AutoConfig.from_pretrained("google/flan-t5-xl")
        self.config.use_cache = False  # Disable caching mechanism
        # Optionally reduce attention settings
        # config.num_layers = config.num_layers // 2  # Reduce layers if needed
        # config.num_heads = 8  # Reduce attention heads

    # 5. Progressive loading function
    def load_model_progressively(self):
        try:
            # First attempt: Basic loading with mixed precision
            print("Attempting basic mixed precision loading...")
            config = self.config
            model = TFAutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-xl",
                config=config
            )
            return model
        except Exception as e:
            print(f"First attempt failed: {e}")

            try:
                # Second attempt: Load on CPU first
                print("Attempting CPU loading...")
                with tf.device('/CPU:0'):
                    model = TFAutoModelForSeq2SeqLM.from_pretrained(
                        "google/flan-t5-xl",
                        config=self.config,
                        from_pt=True,
                        low_cpu_mem_usage=True
                    )
                return model
            except Exception as e:
                print(f"Second attempt failed: {e}")

                try:
                    # Third attempt: With minimal config and gradual loading
                    print("Attempting minimal config loading...")
                    # Clear existing GPU memory
                    tf.keras.backend.clear_session()
                    gc.collect()

                    # Load in parts
                    with tf.device('/CPU:0'):
                        model = TFAutoModelForSeq2SeqLM.from_pretrained(
                            "google/flan-t5-xl",
                            config=config,
                            from_pt=True,
                            low_cpu_mem_usage=True,
                            load_weights_only=True
                        )
                    return model
                except Exception as e:
                    print(f"Third attempt failed: {e}")
                    raise Exception("Unable to load model with current memory constraints")

    # 7. Main loading function
    def initialize_model(self):
        try:
            # Clear any existing models/memory
            tf.keras.backend.clear_session()
            gc.collect()

            # Try progressive loading
            model = self.load_model_progressively()

            # Test the model with minimal input
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")

            #test_input = tokenizer("test", return_tensors="tf")
            #_ = model.generate(test_input.input_ids, max_length=5)

            print("Model loaded successfully!")
            return tokenizer, model

        except Exception as e:
            print(f"Final loading attempt failed: {e}")
            print("Consider using a smaller model or different approach")
            return None
