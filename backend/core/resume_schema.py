#/backend/core/resume_schema.py

from pydantic import BaseModel
from typing import List, Optional


class Education(BaseModel):
    degree: str = ""
    university: str = ""
    location: str = ""
    duration: str = ""


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    location: str = ""
    full_description: List[str] = []


class ResumeResponse(BaseModel):
    name: str = ""
    phone: str = ""
    mail: str = ""
    location: str = ""
    linkedin: str = ""
    education: List[Education] = []
    experience: List[Experience] = []
    skills: List[str] = []
