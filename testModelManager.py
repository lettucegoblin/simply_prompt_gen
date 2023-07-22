from modelManager import modelManager
modelLocation = "./models/promptgen-lexart"

def generate(context, model):
    m_manage = modelManager(model)
    output = m_manage.generatePrompt(context)

# get user input
while True:
    context = input("Enter a prompt: ")
    generate(context, modelLocation)