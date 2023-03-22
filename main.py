'''
This repl demos a Q&A bot over the wandb documentation at docs.wandb.ai using LangChain and Openai

# USING Weights & Biases
Weights & Biases allows you to keep track of your experiments, find your best results 
and save your best models. You will need a free account to use Weights & Biases, 
sign up at https://wandb.ai/site

W&B API KEY
- When a W&B run starts, you will be prompted for your API key, which 
- you can find here: https://wandb.ai/authorize
- We recommend adding it to your Replit Secrets and naming it "WANDB_API_KEY"
- You can also set an environment varibale for your API key in your script
- like so: os.environ["WANDB_API_KEY"] = XXX

Openai API KEY
- Signup at https://beta.openai.com/signup
- create or use an existing key from : https://platform.openai.com/account/api-keys
'''


import wandb
from langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.prompts import PromptTemplate
import faiss
import gradio as gr
import pickle

gr.close_all()

PROJECT = "wandb_docs_bot"

run = wandb.init(project=PROJECT)

def validate_openai_key(api_key):
  if api_key and api_key.startswith("sk-") and len(api_key) > 50:
      return True
  else:
      return False

def load_vectostore():
  artifact = run.use_artifact('darek/wandb_docs_bot/faiss_store:latest',
                              type='search_index')
  artifact_dir = artifact.download()
  index = faiss.read_index(artifact_dir + "/docs.index")

  with open(artifact_dir + "/faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
  store.index = index
  return store


def load_prompt():
  dataset_artifact_path = 'parambharat/wandb_docs_bot/docs_dataset:latest'
  artifact = run.use_artifact(dataset_artifact_path, type='dataset')
  artifact_path = artifact.get_path("combine_prompt.txt")
  file = artifact_path.download()
  prompt_template = (open(file, "r").read())
  prompt = PromptTemplate(input_variables=["question", "summaries"],
                          template=prompt_template)
  return prompt



def load_chain(openai_api_key):
    vectorstore = load_vectostore()
    prompt = load_prompt()
    chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        vectorstore=vectorstore,
        combine_prompt=prompt)
    return chain


def get_answer(question, chain):
  if chain is not None:
    result = chain(
      {
        "question": question,
      },
      return_only_outputs=True,
    )
    response = f"Answer:\t{result['answer']}\n\nSources:\t{result['sources']}\n"
    return response




class Chat:

  def __init__(self):
    self.chain = None

  def __call__(self, message, history, openai_api_key):
    if self.chain is None:
      self.chain = load_chain(openai_api_key)

    history = history or []
    message = message.lower()
    response = get_answer(message, self.chain)
    if response is None:
      response = "Please enter a valid Openai API Key and try again. "
    history.append((message, response))
    return history, history


with gr.Blocks() as demo:
  gr.HTML(
    """<div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Wandb QandA Bot
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Hi, I'm a wandb documentaion Q and A bot, start by typing in your OpenAI API key, questions/issues you have related to wandb usage and then press enter.<br>
        Built using <a href="https://langchain.readthedocs.io/en/latest/" target="_blank">LangChain</a> and <a href="https://github.com/gradio-app/gradio" target="_blank">Gradio Github repo</a>
        </p>
    </div>""")
  with gr.Row():
    question = gr.Textbox(
      label='Type in your questions about wandb here and press Enter!',
      placeholder='How do i log images with wandb ?')
    openai_api_key = gr.Textbox(type='password',
                                label="Enter your OpenAI API key here",)
  state = gr.State()
  chatbot = gr.Chatbot()
  question.submit(Chat(), [question, state, openai_api_key], [chatbot, state])

if __name__ == "__main__":
  demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8086, show_error=True)
