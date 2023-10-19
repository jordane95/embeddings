"""
openai==0.26.4
tiktoken==0.2.0
"""
import argparse
import logging
import os
import pathlib
import pickle

import openai
import tiktoken
import random

logging.basicConfig(level=logging.INFO)


API_KEY = "YOUR_KEY"


class OpenAIEmbedder:
    """
    Benchmark OpenAIs embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=32, save_emb=False, **kwargs):
        self.engine = engine # 'text-embedding-ada-002'
        self.max_token_len = 8191
        self.batch_size = batch_size
        self.save_emb = save_emb # Problematic as the filenames may end up being the same
        self.base_path = f"embeddings/{engine.split('/')[-1]}/"
        self.tokenizer = tiktoken.encoding_for_model(engine)
        self.task_name = task_name

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def encode(
        self, 
        sentences,
        decode=True,
        idx=None,
        **kwargs
    ):

        openai.api_key = API_KEY

        fin_embeddings = []

        embedding_path = f"{self.base_path}/{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]

                all_tokens = []
                used_indices = []
                for j, txt in enumerate(batch):
                    # tokens = self.tokenizer.encode(txt, add_special_tokens=False)
                    if not(txt):
                        print("Detected empty item, which is not allowed by the OpenAI API - Replacing with empty space")
                        txt = " "
                    tokens = self.tokenizer.encode(txt)
                    token_len = len(tokens)
                    if token_len > self.max_token_len:
                        tokens = tokens[:self.max_token_len]
                    # For some characters the API raises weird errors, e.g. input=[[126]]
                    if decode:
                        tokens = self.tokenizer.decode(tokens)
                    all_tokens.append(tokens)
                    used_indices.append(j)

                out = [[]] * len(batch)
                if all_tokens:
                    response = openai.Embedding.create(input=all_tokens, model=self.engine)
                    # May want to sleep here to avoid getting too many requests error
                    # time.sleep(1)
                    assert len(response["data"]) == len(
                        all_tokens
                    ), f"Sent {len(all_tokens)}, got {len(response['data'])}"

                    for data in response["data"]:
                        idx = data["index"]
                        # OpenAI seems to return them ordered, but to be save use the index and insert
                        idx = used_indices[idx]
                        embedding = data["embedding"]
                        out[idx] = embedding

                fin_embeddings.extend(out)
        # Save embeddings
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings



def load_data(lang: str):
    data = list()
    prefix = '/data01/lizehan/proqa/data/codesearch'
    filename = f'{prefix}/{lang}/final/jsonl/test/{lang}_test_0.jsonl.gz'
    with gzip.open(filename, 'rt', encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def evaluate(model, data):
    random.seed(42)
    random.shuffle(data)

    text_queries, code_corpus = list(), list()
    for item in data:
        text_queries.append(item['docstring'])
        code_corpus.append(item['code'])
    query_embeddings = model.encode(text_queries)
    corpus_embeddings = model.encode(code_corpus)

    ranks = list()
    for i in range(0, len(data), 1000):
        q_emb = query_embeddings[i: i+1000]
        c_emb = corpus_embeddings[i: i+1000]
        if c_emb.shape[0] < 1000:
            padding = corpus_embeddings[i + c_emb.shape[0] - 1000: i]
            c_emb = np.concatenate((c_emb, padding), axis=0)
        sim_mat = cos_sim(q_emb, c_emb)
        for j in range(q_emb.shape[0]):
            correct_score = sim_mat[j, j]
            rank = torch.sum(sim_mat[j] >= correct_score).item()
            ranks.append(rank)
    
    result = {
        "mrr@1000": torch.mean(1 / torch.tensor(ranks)).item()
    }
    return result


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="text-embedding-ada-002")
    parser.add_argument("--taskname", type=str, default='codesearch')
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):

    # There are two different batch sizes
    # OpenAIEmbedder(...) batch size arg is used to send X embeddings to the API
    # evaluation.run(...) batch size arg is how much will be saved / pickle file (as it's the total sent to the embed function)

    langs = ['ruby', 'go', 'java', 'javascript', 'php', 'python']

    scores = 0

    for lang in langs:
        model = OpenAIEmbedder(args.engine, task_name=f"{args.taskname}-{lang}", batch_size=args.batchsize, save_emb=True)
        data = load_data(lang)
        result = evaluate(model, data)
        with open(os.path.join(args.output_dir, lang + '.json'), 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"{lang}: {json.dumps(result, indent=2)}")
        scores += result['mrr@1000']
    
    with open(os.path.join(args.output_dir, "avg.json"), 'w') as f:
        json.dump({"mrr@1000": scores / len(langs)}, f, indent=2)
    logger.info(f"average score: {scores/len(langs)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

