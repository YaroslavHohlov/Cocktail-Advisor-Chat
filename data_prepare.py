import pandas as pd
import json
import os
from ast import literal_eval

class DatasetLoader:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.drinks_df = None

    def load_dataset(self):

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at path: {self.dataset_path}")
        
        self.drinks_df = pd.read_csv(self.dataset_path)
        print(f"Dataset successfully loaded. Number of records: {len(self.drinks_df)}")

    def preprocess_dataset(self):
        """
        Preprocessing the dataset:
        - Handling missing values
        - Converting the ingredients and ingredientMeasures columns to lists
        - Combining ingredients and their measures
        - Normalizing the data
        """
        if self.drinks_df is None:
            raise ValueError("The dataset is not loaded. Call load_dataset() first.")

        self.drinks_df.fillna("", inplace=True)

        def parse_list_column(value):
            if isinstance(value, str) and value:
                try:
                    return literal_eval(value)
                except:
                    return []
            return []

        self.drinks_df["ingredients"] = self.drinks_df["ingredients"].apply(parse_list_column)
        self.drinks_df["ingredientMeasures"] = self.drinks_df["ingredientMeasures"].apply(parse_list_column)

        # Combining ingredients and their measurements into a single list
        def combine_ingredients_and_measures(row):
            ingredients = row["ingredients"]
            measures = row["ingredientMeasures"]
            combined = []
            for i in range(len(ingredients)):
                ingredient = ingredients[i] if ingredients[i] is not None else ""
                measure = measures[i] if i < len(measures) and measures[i] is not None else ""
                combined.append({
                    "ingredient": ingredient.strip() if isinstance(ingredient, str) else "",
                    "measure": measure.strip() if isinstance(measure, str) else ""
                })
            return combined

        self.drinks_df["combined_ingredients"] = self.drinks_df.apply(combine_ingredients_and_measures, axis=1)

        self.drinks_df.drop(columns=["ingredients", "ingredientMeasures"], inplace=True)

        self.drinks_df["name"] = self.drinks_df["name"].str.lower()
        self.drinks_df["category"] = self.drinks_df["category"].str.lower()
        self.drinks_df["glassType"] = self.drinks_df["glassType"].str.lower()
        self.drinks_df["instructions"] = self.drinks_df["instructions"].str.lower()

        print("The dataset was successfully processed.")

    def filter_by_ingredient(self, ingredient):
        """
        Filter drinks by ingredient.

        :param ingredient: Name of the ingredient to filter
        :return: List of drinks that contain this ingredient
        """
        if self.drinks_df is None:
            raise ValueError("The dataset is not loaded. Call load_dataset() first.")

        ingredient = ingredient.lower()
        filtered_drinks = self.drinks_df[
            self.drinks_df["combined_ingredients"].apply(
                lambda x: any(ingredient in ing["ingredient"].lower() for ing in x)
            )
        ]
        return filtered_drinks.to_dict(orient="records")

    def filter_by_alcoholic(self, is_alcoholic):
        """
        Filter drinks by alcohol content.

        :param is_alcoholic: True for alcoholic, False for non-alcoholic
        :return: List of drinks in the corresponding category
        """
        if self.drinks_df is None:
            raise ValueError("The dataset is not loaded. Call load_dataset() first.")

        filtered_drinks = self.drinks_df[
            self.drinks_df["alcoholic"].str.lower() == ("alcoholic" if is_alcoholic else "non alcoholic")
        ]
        return filtered_drinks.to_dict(orient="records")

    def save_processed_dataset(self, output_path):
        """
        Save the processed dataset to a JSON file.

        :param output_path: Path to save the JSON file
        """
        if self.drinks_df is None:
            raise ValueError("The dataset is not loaded. Call load_dataset() first.")

        self.drinks_df.to_json(output_path, orient="records", indent=4)
        print(f"The processed dataset is saved to the path: {output_path}")


if __name__ == "__main__":
    dataset_path = "drinks.csv"
    loader = DatasetLoader(dataset_path)

    try:
        loader.load_dataset()

        print("Dataset columns:", list(loader.drinks_df.columns))
        loader.preprocess_dataset()
        lemon_drinks = loader.filter_by_ingredient("lemon")
        print(f"Drinks with lemon: {len(lemon_drinks)}")
        for drink in lemon_drinks[:5]:
            print(f"- {drink['name']}")

        non_alcoholic_drinks = loader.filter_by_alcoholic(False)
        print(f"Soft drinks: {len(non_alcoholic_drinks)}")

        loader.save_processed_dataset("processed_drinks.json")

    except Exception as e:
        print(f"Error: {e}")