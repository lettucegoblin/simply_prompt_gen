import gradio as gr
import os
from modelManager import modelManager
modelDirectory = "./models/"

def promptgen(context, model):
    print(context, model, modelDict[model])
    # load model
    m_manage = modelManager(modelDict[model])
    output = m_manage.generatePrompt(context)
    m_manage.unloadModel()
    m_manage = None
    return output

modelList = os.listdir(modelDirectory)
# create a dictionary of model names and model paths
modelDict = {}
for model in modelList:
    modelDict[model] = modelDirectory + model
# create a dropdown of model names
default_model = next(iter(modelDict)) if modelDict else None


with gr.Blocks() as PromptBlock:
    prompt = gr.Textbox(label="Prompt")
    
    output = gr.Textbox(label="Output Box").style(show_copy_button=True)
    gen_btn = gr.Button("Generate")
    modelDropdown = gr.Dropdown(choices=list(modelDict.keys()), label="Model", value=default_model)
    prompt.submit(fn=promptgen, inputs=[prompt, modelDropdown], outputs=output, api_name="promptgen")
    gen_btn.click(fn=promptgen, inputs=[prompt, modelDropdown], outputs=output, api_name="promptgen")

PromptBlock.launch(server_name="0.0.0.0", server_port=6969, share=False)