# Cocktail Advisor Chat

## Overview

This project is a chat application for cocktail recommendations, utilizing the local `distilgpt2` model to process user queries. The system integrates a vector database (FAISS) for finding similar cocktails and supports a REST API for interaction with a web interface. The project is based on a cocktail recipe dataset from Kaggle.

---

## Project Structure

- `LLM.py` - Handles user queries using `distilgpt2`.
- `FAISS_integrate.py` - Implements the FAISS vector database for finding similar cocktails.
- `api.py` - Provides a REST API for interaction with the web interface.
- `templates/index.html` - HTML page for the chat interface.
- `processed_drinks.json` - Processed dataset with cocktail recipes.
- `faiss_index` - FAISS index for the vector database.
- `faiss_metadata.pkl` - Metadata for the vector database.
- `user_preferences.json` - File storing user preferences.
- `environment.yaml` - Conda environment file with project dependencies.

---

## File Descriptions

- **`LLM.py`**: Processes user queries (searching cocktails by ingredients, filtering by alcohol content, recommendations) using `distilgpt2`.
- **`FAISS_integrate.py`**: Creates and uses FAISS for finding similar cocktails with embeddings (`all-MiniLM-L6-v2`).
- **`api.py`**: Implements a REST API (endpoints: `/query`, `/preferences`) for interaction with the web interface.
- **`templates/index.html`**: HTML page for entering queries and displaying responses from the API.
- **`environment.yaml`**: Defines the Conda environment with all required dependencies.

---

## Achievements

- Successfully processes queries for searching cocktails by ingredients (e.g., "What are the 5 cocktails containing lemon?").
- Successfully processes queries for recommending similar cocktails (e.g., "Recommend a cocktail similar to Hot Creamy Bush").
- The system detects and saves user preferences (e.g., "I like gin and tonic").
- FAISS is created and used for finding similar cocktails.
- REST API and web interface are implemented for system interaction.

---

## Limitations and Unresolved Issues

- Incorrect filtering of non-alcoholic cocktails (returns alcoholic cocktails).
- Incorrect recommendations based on favorite ingredients (no cocktails returned).
- Unstructured `distilgpt2` responses for RAG (repetitions, unclear answers).
- "Internal Server Error" (500) for some queries due to issues in `LLM.py`.
- Incomplete OpenAI integration (`LLM_vers2.py`) due to API quota exceeded.

---

## Future Improvements

- Fix filtering of non-alcoholic cocktails and recommendations based on favorite ingredients.
- Improve RAG by tuning prompts for `distilgpt2` or using another model.
- Replenish OpenAI balance and complete integration of `LLM_vers2.py`.
- Enhance the web interface (add styles, error handling).

---

## User Guide

### Setting Up the Environment
1. **Install Miniconda or Anaconda**:
   - Download and install from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Create the Conda environment**:
   - Use the provided `environment.yaml` to create the environment:
     ```bash
     conda env create -f environment.yaml
     ```
   - Activate the environment:
     ```bash
     conda activate cocktail-advisor
     ```
3. **Verify dependencies**:
   - Ensure all required packages are installed (e.g., `fastapi`, `uvicorn`, `transformers`, `sentence-transformers`, `faiss-cpu`).

### Running the Application
1. **Run the API**:
   - Start the API server:
     ```bash
     uvicorn api:app --reload
     ```
   - The API will be available at `http://127.0.0.1:8000`.
2. **Access the web interface**:
   - Open `templates/index.html` in a browser (e.g., `file:///path_to_folder/index.html`).
   - Ensure the API is running to handle queries.
3. **Test queries**:
   - Enter a query in the input field (e.g., "What are the 5 cocktails containing lemon?") and click "Send".
   - Responses will be displayed below the input field.

### Example Queries
- "What are the 5 cocktails containing lemon?" - Returns 5 cocktails with lemon.
- "Recommend a cocktail similar to Hot Creamy Bush" - Returns similar cocktails.
- "I like gin and tonic" - Detects and saves "gin" as a favorite ingredient.

### Notes
- Some queries may return incorrect results (e.g., non-alcoholic cocktails, favorite ingredient recommendations) due to unresolved issues.
- The web interface may display errors if the API is not running or encounters issues.