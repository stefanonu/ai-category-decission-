from pydantic import BaseModel
from typing import Dict

# Define a Pydantic model to represent the category scores
class CategoryScores(BaseModel):
    social_interaction: int = 0
    language: int = 0
    economy: int = 0
    policy: int = 0
    education: int = 0
    art: int = 0
    health: int = 0
    family: int = 0
    media: int = 0
    philosophy: int = 0
    age: int = 0
    technology: int = 0
    national: int = 0
    culture: int = 0
    environment: int = 0
    law: int = 0
    gender: int = 0
    history: int = 0
    psychology: int = 0
    religion: int = 0
    inequality: int = 0
    urban: int = 0
    food: int = 0
    sport: int = 0
    travel: int = 0
    war: int = 0
    cities_countries: int = 0
    labour: int = 0
