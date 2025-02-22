import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from FAISS_integrate import VectorDB

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class CocktailAssistant:
    def __init__(self, vector_db_path="faiss_index", metadata_path="faiss_metadata.pkl", 
                 json_path="processed_drinks.json", user_preferences_path="user_preferences.json",
                 openai_api_key=openai_api_key):
        """
        Initializing the Cocktail Assistant with OpenAI.

        :param vector_db_path: Path to the FAISS index
        :param metadata_path: Path to the FAISS metadata
        :param json_path: Path to the processed JSON with drink data
        :param user_preferences_path: Path to the user preferences file
        :param openai_api_key: OpenAI API key
        """
        self.vector_db = VectorDB(index_path=vector_db_path, metadata_path=metadata_path)
        self.vector_db.load_index()
        self.drinks_data = self.load_json(json_path)

        self.user_preferences_path = user_preferences_path
        self.user_preferences = self.load_user_preferences()

        if openai_api_key:
            self.llm = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("OpenAI API key is required for LLM initialization.")

    def load_json(self, json_path):
        """
        Load processed data from a JSON file.

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
        Loading user preferences with JSON file.

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
        return filtered_drinks[:limit]

    def get_cocktails_by_alcoholic(self, is_alcoholic, ingredient=None, limit=5):
        alcoholic_status = "alcoholic" if is_alcoholic else "non alcoholic"
        filtered_drinks = [drink for drink in self.drinks_data 
                        if drink["alcoholic"].lower() == alcoholic_status]
        
        if ingredient:
            ingredient = ingredient.lower()
            filtered_drinks = [drink for drink in filtered_drinks 
                            if any(ingredient in ing["ingredient"].lower() 
                                    for ing in drink["combined_ingredients"])]
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
        return similar_drinks

    def recommend_by_preferences(self, k=5):
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

            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cocktail advisor assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            llm_response = response.choices[0].message.content
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