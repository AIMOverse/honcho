from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.db import SessionLocal
from src.models import UserPersona, Demographics, HabitsPermissions, EmotionalState, MBTIPersonality, StressLevel
from src.schemas import UserPersona, PersonaRequest, PersonaResponse

router = APIRouter()

@router.post("/survey", response_model=PersonaResponse)
async def create_survey(survey: PersonaRequest, db: AsyncSession = Depends(SessionLocal)):
    demographics = Demographics(**survey.demographics.dict())
    habits_permissions = HabitsPermissions(**survey.habits_permissions.dict())
    emotional_state = EmotionalState(**survey.emotional_state.dict())
    personality_assessment = MBTIPersonality(**survey.personality_assessment.dict())
    stress_assessment = StressLevel(**survey.stress_assessment.dict())

    db.add_all([demographics, habits_permissions, emotional_state, personality_assessment, stress_assessment])
    await db.commit()

    user_personas = UserPersona(
        username= UserPersona.username,
        demographics=demographics,
        habits_permissions=habits_permissions,
        emotional_state=emotional_state,
        personality_assessment=personality_assessment,
        stress_assessment=stress_assessment,
    )
    db.add(user_personas)
    await db.commit()

    return PersonaResponse(success=True, message="Survey created successfully", processed_data=survey.dict())