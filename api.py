import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from LLM import CocktailAssistant 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    logger.info("Initializing CocktailAssistant...")
    assistant = CocktailAssistant()
    logger.info("CocktailAssistant initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize CocktailAssistant: {str(e)}", exc_info=True)
    raise

class QueryRequest(BaseModel):
    query: str

class PreferencesRequest(BaseModel):
    favorite_ingredients: list[str] = []
    favorite_cocktails: list[str] = []

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Processing a user request.

    :param request: Request with the request text
    :return: System response
    """
    logger.info(f"Received query: {request.query}")
    try:
        response = assistant.process_query(request.query)
        logger.info(f"Query processed successfully: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)  # Додаємо traceback
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preferences")
async def get_preferences():
    """
    Retrieve user preferences.

    :return: Preferences dictionary
    """
    logger.info("Fetching user preferences...")
    try:
        preferences = assistant.user_preferences
        logger.info(f"Preferences fetched: {preferences}")
        return preferences
    except Exception as e:
        logger.error(f"Error fetching preferences: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preferences")
async def update_preferences(request: PreferencesRequest):
    """
    Updating user preferences.

    :param request: Request with new preferences
    :return: Message about successful update
    """
    logger.info(f"Updating preferences: {request}")
    try:
        assistant.user_preferences["favorite_ingredients"] = request.favorite_ingredients
        assistant.user_preferences["favorite_cocktails"] = request.favorite_cocktails
        assistant.save_user_preferences()
        logger.info("Preferences updated successfully.")
        return {"message": "Preferences updated successfully"}
    except Exception as e:
        logger.error(f"Error updating preferences: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))