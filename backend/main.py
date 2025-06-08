# /backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router
import uvicorn

app = FastAPI(title="Resume Extractor API", version="1.0.0")

# Enable CORS for public
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# # Serve public files
# app.mount("/", StaticFiles(directory="public", html=True), name="public")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}



@app.get("/")
async def root():
    return {
        "message": "Resume Extractor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)