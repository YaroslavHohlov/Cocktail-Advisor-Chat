import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import json

class VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path="faiss_index", metadata_path="faiss_metadata.pkl"):
        """
        Initialize the FAISS vector database.

        :param model_name: Name of the model for text embedding (sentence-transformers)
        :param index_path: Path to store the FAISS index
        :param metadata_path: Path to store metadata (drink information)
        """
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []  # List for storing metadata (drink information)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def load_from_json(self, json_path):
        """
        Loading processed data from a JSON file.

        :param json_path: Path to the JSON file with processed data
        :return: List of dictionaries with information about drinks
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at path: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            drinks_data = json.load(f)
        print(f"Data successfully loaded from JSON. Number of records: {len(drinks_data)}")
        return drinks_data

    def create_index(self, drinks_data):
        """
        Creating a FAISS index from drinks data.

        :param drinks_data: List of dictionaries with drinks information (from JSON)
        """
        #We create text descriptions for each drink (e.g. name + ingredients + instructions)
        texts = []
        self.metadata = []
        for drink in drinks_data:
            if "combined_ingredients" not in drink:
                print(f"Warning: 'combined_ingredients' is missing from the entry {drink.get('name', 'unknown')}")
                continue
            
            ingredients_text = " ".join([ing["ingredient"] for ing in drink["combined_ingredients"]])
            description = f"{drink['name']} {ingredients_text} {drink['instructions']}"
            texts.append(description)
            self.metadata.append(drink)

        if not texts:
            raise ValueError("Unable to create descriptions for drinks. Please check your data.")

        # Converting texts into vectors
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Initialize FAISS index (we use IndexFlatL2 for simplicity)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        print(f"FAISS індекс створено. Кількість записів: {self.index.ntotal}")

        self.save_index()

    def save_index(self):
        """
        Saving FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"FAISS index and metadata saved: {self.index_path}, {self.metadata_path}")

    def load_index(self):
        """
        Loading FAISS index and metadata from disk.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"FAISS index and metadata loaded. Number of records: {self.index.ntotal}")
        else:
            raise FileNotFoundError("No index or metadata files were found.")

    def search_similar_drinks(self, query, k=5):
        """
        Search for similar drinks by query.

        :param query: Text query (e.g. ingredients or drink name)
        :param k: Number of results to return
        :return: List of similar drinks
        """
        if self.index is None:
            raise ValueError("FAISS індекс не створено або не завантажено.")

        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")


        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["similarity_score"] = float(distance)
                results.append(result)

        return results


if __name__ == "__main__":
    json_path = "processed_drinks.json"
    vector_db = VectorDB()

    try:
        drinks_data = vector_db.load_from_json(json_path)
        vector_db.create_index(drinks_data)
        query = "cocktail with gin and lemon juice"
        similar_drinks = vector_db.search_similar_drinks(query, k=5)
        print(f"Схожі напої для запиту '{query}':")
        for drink in similar_drinks:
            print(f"- {drink['name']} (similarity score: {drink['similarity_score']})")

    except Exception as e:
        print(f"Error: {e}")