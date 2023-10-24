from dotenv import find_dotenv, load_dotenv

from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI

load_dotenv(find_dotenv())




#img to text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text

#llm for short story

def generate_story(scenario):
    template =  """
    you are a storyteller; 
    you can generate a short story based on a simple narrative, the story should be no more than 20 words
    
    CONTEXT : {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template=template, input_variables = ['scenario'])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature = 1), prompt=promt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)

    print(story)
    return(story)

#TTS



scenario = img2text("BG1.png")
story = generate_story(scenario)