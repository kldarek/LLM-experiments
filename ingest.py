import os
import wandb
import faiss
import pickle
import json
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

PROJECT = "wandb_docs_bot"

run = wandb.init(project=PROJECT)


def download_raw_dataset():
    dataset_artifact_path = 'parambharat/wandb_docs_bot/docs_dataset:latest'
    artifact = run.use_artifact(dataset_artifact_path, type='dataset')
    artifact_path = artifact.get_path("wandb_docs.json")
    file = artifact_path.download()
    return file


def load_documents(fname):
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for line in open(fname, "r"):
        line = json.loads(line)
        for chunk in splitter.split_text(line["reference"]):
            source_chunks.append(Document(page_content=chunk, metadata={"source": line["source"]}))
    return source_chunks


def create_and_save_index(documents):
    store = FAISS.from_documents(documents,OpenAIEmbeddings())
    artifact = wandb.Artifact("faiss_store", type="search_index")
    faiss.write_index(store.index, "docs.index")
    artifact.add_file("docs.index")
    store.index = None
    with artifact.new_file("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    wandb.log_artifact(artifact, "docs_index", type="embeddings_index")
    return store

def main():
    dataset_path = download_raw_dataset()
    documents = load_documents(dataset_path)
    store = create_and_save_index(documents)
    
if __name__ == "__main__":
    main()