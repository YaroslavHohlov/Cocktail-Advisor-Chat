import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import json
import os
from FAISS_integrate import VectorDB

class CocktailAssistant:
    def __init__(self, llm_model="distilgpt2", vector_db_path="faiss_index", 
                 metadata_path="faiss_metadata.pkl", json_path="processed_drinks.json",
                 user_preferences_path="user_preferences.json"):
        """
        Initialize the cocktail assistant.

        :param llm_model: LLM model name (e.g. distilgpt2)
        :param vector_db_path: Path to FAISS index
        :param metadata_path: Path to FAISS metadata
        :param json_path: Path to processed JSON with drink data
        :param user_preferences_path: Path to user preferences file
        """
        self.llm = pipeline("text-generation", model=llm_model, max_length=200, truncation=True)
        
        self.vector_db = VectorDB(index_path=vector_db_path, metadata_path=metadata_path)
        self.vector_db.load_index()
        
        self.drinks_data = self.load_json(json_path)

        self.user_preferences_path = user_preferences_path
        self.user_preferences = self.load_user_preferences()

    def load_json(self, json_path):
        """
        Loading processed data from a JSON file.

        :param json_path: Path to the JSON file with processed data
        :return: List of dictionaries with information about drinks
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at path: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            drinks_data = json.load(f)
        print(f"Loaded drinks data. Number of records: {len(drinks_data)}")
        return drinks_data

    def load_user_preferences(self):
        """
        Loading user preferences from JSON file.

        :return: Dictionary with user preferences
        """
        if os.path.exists(self.user_preferences_path):
            with open(self.user_preferences_path, "r", encoding="utf-8") as f:
                preferences = json.load(f)
            print("User preferences loaded.")
        else:
            preferences = {"favorite_ingredients": [], "favorite_cocktails": []}
            print("No user preferences found. Initialized empty preferences.")
        return preferences

    def save_user_preferences(self):
        """
        Saving user preferences to a JSON file.
        """
        with open(self.user_preferences_path, "w", encoding="utf-8") as f:
            json.dump(self.user_preferences, f, indent=4)
        print("User preferences saved.")

    def detect_user_preferences(self, query):
        """
        Detect user preferences from query text.

        :param query: User query text
        """
        query_lower = query.lower()
        if "my favorite ingredient" in query_lower or "i like" in query_lower:
            for drink in self.drinks_data:
                for ing in drink["combined_ingredients"]:
                    ingredient = ing["ingredient"].lower()
                    if ingredient in query_lower and ingredient not in self.user_preferences["favorite_ingredients"]:
                        self.user_preferences["favorite_ingredients"].append(ingredient)
                        print(f"Added {ingredient} to favorite ingredients.")
            self.save_user_preferences()

    def get_cocktails_by_ingredient(self, ingredient, limit=5):
        """
        Search for cocktails by ingredient.

        :param ingredient: Ingredient name
        :param limit: Maximum number of results
        :return: List of cocktails
        """
        ingredient = ingredient.lower()
        filtered_drinks = [drink for drink in self.drinks_data 
                          if any(ingredient in ing["ingredient"].lower() 
                                for ing in drink["combined_ingredients"])]
        print(f"Filtered drinks with ingredient {ingredient}: {[drink['name'] for drink in filtered_drinks]}")
        return filtered_drinks[:limit]

    def get_cocktails_by_alcoholic(self, is_alcoholic, ingredient=None, limit=5):
        """
        Search for cocktails by alcohol content and ingredient.

        :param is_alcoholic: True for alcoholic, False for non-alcoholic
        :param ingredient: Ingredient name (optional)
        :param limit: Maximum number of results
        :return: List of cocktails
        """
        alcoholic_status = "alcoholic" if is_alcoholic else "non alcoholic"
        filtered_drinks = [drink for drink in self.drinks_data 
                          if drink["alcoholic"].lower() == alcoholic_status]
        print(f"Filtered drinks (alcoholic={is_alcoholic}): {[drink['name'] for drink in filtered_drinks]}")
        
        if ingredient:
            ingredient = ingredient.lower()
            filtered_drinks = [drink for drink in filtered_drinks 
                              if any(ingredient in ing["ingredient"].lower() 
                                    for ing in drink["combined_ingredients"])]
        print(f"Filtered drinks with ingredient {ingredient}: {[drink['name'] for drink in filtered_drinks]}")
        return filtered_drinks[:limit]

    def recommend_similar_cocktail(self, cocktail_name, k=5):
        """
        Recommend similar cocktails based on cocktail name.

        :param cocktail_name: Cocktail name
        :param k: Number of recommendations
        :return: List of similar cocktails
        """
        query = cocktail_name.lower()
        similar_drinks = self.vector_db.search_similar_drinks(query, k=k)
        print(f"Found similar cocktails for {cocktail_name}: {[drink['name'] for drink in similar_drinks]}")
        return similar_drinks

    def recommend_by_preferences(self, k=5):
        """
        Cocktail recommendation based on user preferences.

        :param k: Number of recommendations
        :return: List of cocktails
        """
        print(f"Favorite ingredients: {self.user_preferences['favorite_ingredients']}")
        if not self.user_preferences["favorite_ingredients"]:
            return []
        query = " ".join(self.user_preferences["favorite_ingredients"])
        print(f"Query for vector DB: {query}")
        similar_drinks = self.vector_db.search_similar_drinks(query, k=k)
        print(f"Found cocktails: {[drink['name'] for drink in similar_drinks]}")
        return similar_drinks

    def generate_answer(self, query):
        """
        Generate a response to a user query.

        :param query: Query text
        :return: Response
        """
        self.detect_user_preferences(query)
        query_lower = query.lower()

        if "cocktails containing" in query_lower:
            for drink in self.drinks_data:
                for ing in drink["combined_ingredients"]:
                    ingredient = ing["ingredient"].lower()
                    if ingredient in query_lower:
                        cocktails = self.get_cocktails_by_ingredient(ingredient)
                        if cocktails:
                            response = f"Here are 5 cocktails containing {ingredient}:\n"
                            for drink in cocktails:
                                response += f"- {drink['name']}\n"
                            return response
                        else:
                            return f"No cocktails found containing {ingredient}."

        elif "non-alcoholic cocktails" in query_lower:
            for drink in self.drinks_data:
                for ing in drink["combined_ingredients"]:
                    ingredient = ing["ingredient"].lower()
                    if ingredient in query_lower:
                        cocktails = self.get_cocktails_by_alcoholic(False, ingredient)
                        if cocktails:
                            response = f"Here are 5 non-alcoholic cocktails containing {ingredient}:\n"
                            for drink in cocktails:
                                response += f"- {drink['name']}\n"
                            return response
                        else:
                            return f"No non-alcoholic cocktails found containing {ingredient}."

        elif "my favourite ingredients" in query_lower:
            if self.user_preferences["favorite_ingredients"]:
                return f"Your favorite ingredients are: {', '.join(self.user_preferences['favorite_ingredients'])}"
            else:
                return "You haven't shared your favorite ingredients yet."

        elif "recommend" in query_lower and "my favourite ingredients" in query_lower:
            cocktails = self.recommend_by_preferences()
            if cocktails:
                response = "Here are 5 cocktails based on your favorite ingredients:\n"
                for drink in cocktails:
                    response += f"- {drink['name']} (similarity score: {drink['similarity_score']})\n"
                return response
            else:
                return "No recommendations found. Please share your favorite ingredients first."

        elif "recommend" in query_lower and "similar to" in query_lower:
            cocktail_name = query_lower.split("similar to")[-1].strip()
            cocktails = self.recommend_similar_cocktail(cocktail_name)
            if cocktails:
                response = f"Here are 5 cocktails similar to {cocktail_name}:\n"
                for drink in cocktails:
                    response += f"- {drink['name']} (similarity score: {drink['similarity_score']})\n"
                return response
            else:
                return f"No similar cocktails found for {cocktail_name}."

        else:
            similar_drinks = self.vector_db.search_similar_drinks(query, k=3)
            context = ""
            if similar_drinks:
                context = "Relevant cocktails found:\n"
                for drink in similar_drinks:
                    ingredients = ", ".join([ing["ingredient"] for ing in drink["combined_ingredients"]])
                    context += f"- {drink['name']} (Ingredients: {ingredients}, Instructions: {drink['instructions']})\n"

            prompt = f"User query: {query}\n\nContext: {context}\nAssistant: Based on your query, here are some recommendations:\n{context}\nIf you need more details, feel free to ask!"
            print(f"Prompt for LLM: {prompt}")

            llm_response = self.llm(prompt, max_length=400, num_return_sequences=1, do_sample=False)[0]["generated_text"]
            print(f"Raw LLM response: {llm_response}")
            llm_response = "\n".join(line for line in llm_response.split("\n") if not line.strip().isdigit())
            print(f"Cleaned LLM response: {llm_response}")
            return llm_response

    def process_query(self, query):
        """
        Processing a user request and returning a response.

        :param query: Query text
        :return: Response
        """
        try:
            return self.generate_answer(query)
        except Exception as e:
            return f"Error processing query: {str(e)}"


if __name__ == "__main__":
    assistant = CocktailAssistant()
    queries = [
        "What are the 5 cocktails containing lemon?",
        "What are the 5 non-alcoholic cocktails containing sugar?",
        "What are my favourite ingredients?",
        "Recommend 5 cocktails that contain my favourite ingredients",
        "Recommend a cocktail similar to Hot Creamy Bush",
        "I like gin and tonic, what can you recommend?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = assistant.process_query(query)
        print(f"Response: {response}")