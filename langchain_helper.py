from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from secret_key import openapi_key
import os

os.environ["OPENAI_API_KEY"] = openapi_key

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

def generate_restaurant_name_and_items(cuisine):

    # Chain 1: Generate Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    name_chain = prompt_template_name | llm | StrOutputParser()

    restaurant_name = name_chain.invoke({
        "cuisine": cuisine
    })

    # Chain 2: Generate Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma separated string."
    )

    food_items_chain = prompt_template_items | llm | StrOutputParser()

    menu_items = food_items_chain.invoke({
        "restaurant_name": restaurant_name
    })

    response = {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))