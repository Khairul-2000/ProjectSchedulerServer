from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, List
from datetime import datetime
import os
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from .services import ProjectService, ProjectInput, Project, projects_store

app = FastAPI(title="Project Management System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class ProjectResponse(BaseModel):
    id: str
    plan: str
    schedule: str
    review: str
    html_output: str
    created_at: datetime

class ProjectListResponse(BaseModel):
    id: str
    project_type: str
    objectives: str
    industry: str
    created_at: datetime

# LangGraph State
class ProjectState(TypedDict):
    input: str
    plan: str
    schedule: str
    review: str
    html_output: str

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

llm = ChatOpenAI(model="gpt-4-turbo", api_key=api_key)

# Agent functions
def planner_agent(state: ProjectState) -> ProjectState:
    project_info = state["input"]
    prompt = f"""
As a Project Planner, your task is to break down the following project description into major phases and detailed tasks. Be very specific and ensure that the output is a clear and concise Markdown list of the phases and their corresponding tasks.

Project Description:
{project_info}

Provide the output as a Markdown list.
"""
    response = llm.invoke(prompt).content
    print("\nPlanner Agent\n")
    print("\nAgent Output: ", response)
    state["plan"] = response
    return state

def scheduler_agent(state: ProjectState) -> ProjectState:
    plan = state["plan"]
    project_info = state["input"]
    prompt = f"""
You are a Project Scheduler. Based on the provided plan and the project description, assign realistic timelines (in weeks) for each task. Assign appropriate team members *only* from the "Team Members" list provided in the project description. Do not create or use any team member names not listed. Assign a project leader *only* from the provided team members, and indicate dependencies where appropriate.

Project Plan:
{plan}

Project Description:
{project_info}

Output as a Markdown table with columns: Task | Duration (weeks) | Team Member | Dependencies
"""
    response = llm.invoke(prompt).content
    print("\nSchedule Agent\n")
    print("\nAgent Output: ", response)
    state["schedule"] = response
    return state

def reviewer_agent(state: ProjectState) -> ProjectState:
    schedule = state["schedule"]
    project_info = state["input"]
    prompt = f"""
You are a Project Reviewer. Review this schedule for completeness, any missing dependencies or tasks, potential bottlenecks, unrealistic timelines, and issues with team member assignments based on the project description.

Here is the schedule to review:
{schedule}

Project Description:
{project_info}

Output suggestions as a Markdown list. If no issues are found that would prevent successful project completion, write: "No significant issues found."
"""
    response = llm.invoke(prompt).content
    print("\nReviewer Agent\n")
    print("\nAgent Output: ", response)
    state["review"] = response
    return state

def html_agent(state: ProjectState) -> ProjectState:
    plan = state["plan"]
    schedule = state["schedule"]
    review = state["review"]
    project_info = state["input"]
    prompt = f"""
You are an HTML Generator. Based on the project plan, schedule, and review, create a single, professional-looking HTML page that summarizes all the information.

IMPORTANT: Convert ALL markdown content to proper HTML format:
- Convert markdown headers (# ## ###) to HTML headers (h1, h2, h3)
- Convert markdown lists (- *) to HTML lists (ul/li)
- Convert markdown tables to proper HTML tables with <table>, <thead>, <tbody>, <tr>, <th>, <td> tags
- Ensure ALL rows from the schedule table are included in the HTML output

Include the following sections in order:
1. **Project Summary:** A brief overview derived from the project description
2. **Project Plan:** Convert the markdown plan to HTML format
3. **Project Schedule:** Convert the COMPLETE markdown table to a properly formatted HTML table with headers and all rows
4. **Review Feedback:** Convert the markdown review to HTML format

Use inline CSS for basic styling (borders for tables, padding, margins). Ensure the output is a complete HTML document with DOCTYPE, html, head, and body tags.

Project Description:
{project_info}

Project Plan (Markdown to convert to HTML):
{plan}

Project Schedule (Markdown table to convert to HTML table - INCLUDE ALL ROWS):
{schedule}

Review Feedback (Markdown to convert to HTML):
{review}

Make sure the HTML table includes:
- Table headers (Task, Duration, Team Member, Dependencies)
- ALL task rows from the markdown table
- Proper table styling with borders and padding
- Responsive layout

Output the complete HTML code.
"""
    response = llm.invoke(prompt).content
    print("\nHTML Generator Agent\n")
    print("\nAgent Output: ", response)
    state["html_output"] = response
    return state

# Create workflow
workflow = StateGraph(ProjectState)
workflow.add_node("planner", planner_agent)
workflow.add_node("scheduler", scheduler_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("html_generator", html_agent)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "scheduler")
workflow.add_edge("scheduler", "reviewer")
workflow.add_edge("reviewer", "html_generator")
workflow.add_edge("html_generator", END)

app_workflow = workflow.compile()

# API Routes
@app.get("/")
async def root():
    return {"message": "Project Management System API with In-Memory Storage"}

@app.post("/generate-project-plan", response_model=ProjectResponse)
async def generate_project_plan(project_input: ProjectInput):
    try:
        # Create project in memory using service
        project = ProjectService.create_project(project_input)
        
        # Format the input for the LangGraph workflow
        formatted_input = f"""
**Project Type:** {project_input.project_type}

**Project Objectives:** {project_input.objectives}

**Industry:** {project_input.industry}

**Team Members:**
{chr(10).join([f"- {member}" for member in project_input.team_members])}

**Project Requirements:**
{chr(10).join([f"- {req}" for req in project_input.requirements])}
"""
        
        # Run the workflow
        output = app_workflow.invoke({"input": formatted_input})
        
        # Update project with generated content using service
        updated_project = ProjectService.update_project_results(
            project.id,
            output["plan"],
            output["schedule"],
            output["review"],
            output["html_output"]
        )
        
        return ProjectResponse(
            id=updated_project.id,
            plan=output["plan"],
            schedule=output["schedule"],
            review=output["review"],
            html_output=output["html_output"],
            created_at=updated_project.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects", response_model=List[ProjectListResponse])
async def get_projects():
    """Get list of all projects"""
    try:
        projects = ProjectService.get_all_projects()
        return [
            ProjectListResponse(
                id=project.id,
                project_type=project.project_type,
                objectives=project.objectives,
                industry=project.industry,
                created_at=project.created_at
            )
            for project in projects
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get a specific project by ID"""
    try:
        project = ProjectService.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectResponse(
            id=project.id,
            plan=project.plan or "",
            schedule=project.schedule or "",
            review=project.review or "",
            html_output=project.html_output or "",
            created_at=project.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a specific project by ID"""
    try:
        if not ProjectService.delete_project(project_id):
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {"message": "Project deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/search/{query}")
async def search_projects(query: str):
    """Search projects by query string"""
    try:
        projects = ProjectService.search_projects(query)
        return [
            ProjectListResponse(
                id=project.id,
                project_type=project.project_type,
                objectives=project.objectives,
                industry=project.industry,
                created_at=project.created_at
            )
            for project in projects
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "storage": "in-memory", "projects_count": len(projects_store)}