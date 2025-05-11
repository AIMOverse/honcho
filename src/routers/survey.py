from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.db import SessionLocal
from src.models import Survey, Demographics, HabitsPermissions, EmotionalState, MBTIPersonality, StressLevel
from src.schemas import SurveyRequest, SurveyResponse

router = APIRouter()

@router.post("/survey", response_model=SurveyResponse)
async def create_survey(survey: SurveyRequest, db: AsyncSession = Depends(SessionLocal)):
    demographics = Demographics(**survey.demographics.dict())
    habits_permissions = HabitsPermissions(**survey.habits_permissions.dict())
    emotional_state = EmotionalState(**survey.emotional_state.dict())
    personality_assessment = MBTIPersonality(**survey.personality_assessment.dict())
    stress_assessment = StressLevel(**survey.stress_assessment.dict())

    db.add_all([demographics, habits_permissions, emotional_state, personality_assessment, stress_assessment])
    await db.commit()

    survey_entry = Survey(
        username=survey.username,
        demographics_id=demographics.id,
        habits_permissions_id=habits_permissions.id,
        emotional_state_id=emotional_state.id,
        personality_assessment_id=personality_assessment.id,
        stress_assessment_id=stress_assessment.id,
    )
    db.add(survey_entry)
    await db.commit()

    return SurveyResponse(success=True, message="Survey created successfully", processed_data=survey.dict())