import pandas as pd
from utils.berts import QuestionAnsweringModel
from random import shuffle


fr_titles = './federal_register_searchKeyword_steel_import_09_24_2020.csv'
fr = pd.read_csv(fr_titles)

titles = fr['title'].to_list()

question1 = "Which country is mentioned?"
question2 = "Which product is mentioned?"

model = QuestionAnsweringModel()

for title in titles:
    try:
        country = model.predict(
            question1, title
        )
        product = model.predict(
            question2, title
        )
        print("------------------")
        print(title)
        print("Country: ", country)
        print("Product: ", product)

    except RuntimeError:
        print(RuntimeError)

if __name__ == "__main__":
    pass
