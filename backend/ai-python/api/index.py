from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Server running on Vercel 🚀"}

@app.get("/test")
def test():
    return {"status": "ok"}