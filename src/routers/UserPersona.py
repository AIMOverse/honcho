"""
Author: Wesley
Date: 2025.5.12

This module defines the API endpoints for managing user personas in the system.
It provides functionality to create a user persona by collecting and storing
demographic, habits, emotional state, personality, and stress-related data.

Endpoints:
- POST /persona: Accepts a PersonaRequest object, processes the data, and stores it in the database.

Dependencies:
- FastAPI for API routing and dependency injection.
- SQLAlchemy for database interactions.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.db import SessionLocal
from src.models import UserPersona, Demographics, HabitsPermissions, EmotionalState, MBTIPersonality, StressLevel
from src.schemas import UserPersona, PersonaRequest, PersonaResponse

# Create a router instance for handling user persona-related endpoints
router = APIRouter()


@router.post("/persona", response_model=PersonaResponse)
async def create_persona(persona: PersonaRequest, db: AsyncSession = Depends(SessionLocal)):
    """
    Create a new user persona based on the provided persona data.

    Parameters:
    - persona (PersonaRequest): The request body containing user persona data.
    - db (AsyncSession): The database session dependency for interacting with the database.

    Workflow:
    1. Extract and map the persona data into individual models (Demographics, HabitsPermissions, etc.).
    2. Add these models to the database session and commit the changes.
    3. Create a UserPersona object linking all the related data.
    4. Add the UserPersona object to the database and commit the changes.

    Returns:
    - PersonaResponse: A response object indicating success or failure, along with the processed data.
    """
    # Map persona data to individual models
    demographics = Demographics(**persona.demographics.dict())
    habits_permissions = HabitsPermissions(**persona.habits_permissions.dict())
    emotional_state = EmotionalState(**persona.emotional_state.dict())
    personality_assessment = MBTIPersonality(**persona.personality_assessment.dict())
    stress_assessment = StressLevel(**persona.stress_assessment.dict())

    # Add the individual models to the database session
    db.add_all([demographics, habits_permissions, emotional_state, personality_assessment, stress_assessment])
    await db.commit()

    # Create a UserPersona object linking all the related data
    user_personas = UserPersona(
        user_id=persona.user_id,
        demographics_id=demographics.id,
        habits_permissions_id=habits_permissions.id,
        emotional_state_id=emotional_state.id,
        personality_assessment_id=personality_assessment.id,
        stress_assessment_id=stress_assessment.id,
    )
    db.add(user_personas)
    await db.commit()

    # Return a success response
    return PersonaResponse(success=True, message="persona created successfully", processed_data=persona.dict())
