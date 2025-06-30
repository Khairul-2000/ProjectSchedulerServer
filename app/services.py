from typing import List, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel

# Define models locally to avoid circular imports
class ProjectInput(BaseModel):
    project_type: str
    objectives: str
    industry: str
    team_members: list[str]
    requirements: list[str]

class Project(BaseModel):
    id: str
    project_type: str
    objectives: str
    industry: str
    team_members: List[str]
    requirements: List[str]
    plan: Optional[str] = None
    schedule: Optional[str] = None
    review: Optional[str] = None
    html_output: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# In-memory storage - this will be shared across the application
projects_store = {}

class ProjectService:
    """Service class for project-related operations using in-memory storage"""
    
    @staticmethod
    def create_project(project_input: ProjectInput) -> Project:
        """Create a new project in memory"""
        project_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        project = Project(
            id=project_id,
            project_type=project_input.project_type,
            objectives=project_input.objectives,
            industry=project_input.industry,
            team_members=project_input.team_members,
            requirements=project_input.requirements,
            created_at=now,
            updated_at=now
        )
        
        projects_store[project_id] = project
        return project
    
    @staticmethod
    def update_project_results(
        project_id: str,
        plan: str,
        schedule: str,
        review: str,
        html_output: str
    ) -> Optional[Project]:
        """Update project with AI-generated results"""
        if project_id not in projects_store:
            return None
            
        project = projects_store[project_id]
        project.plan = plan
        project.schedule = schedule
        project.review = review
        project.html_output = html_output
        project.updated_at = datetime.utcnow()
        
        projects_store[project_id] = project
        return project
    
    @staticmethod
    def get_all_projects() -> List[Project]:
        """Get all projects from memory"""
        return list(projects_store.values())
    
    @staticmethod
    def get_project_by_id(project_id: str) -> Optional[Project]:
        """Get a project by ID from memory"""
        return projects_store.get(project_id)
    
    @staticmethod
    def delete_project(project_id: str) -> bool:
        """Delete a project by ID from memory"""
        if project_id in projects_store:
            del projects_store[project_id]
            return True
        return False
    
    @staticmethod
    def search_projects(query: str) -> List[Project]:
        """Search projects by text in objectives, project_type, or industry"""
        query_lower = query.lower()
        results = []
        
        for project in projects_store.values():
            if (query_lower in project.objectives.lower() or 
                query_lower in project.project_type.lower() or 
                query_lower in project.industry.lower()):
                results.append(project)
        
        return results