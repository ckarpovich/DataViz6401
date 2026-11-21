# course_endpoint.py — the "course-provided REST endpoint" for Week 11
# Instructor: run this BEFORE class:   uvicorn course_endpoint:app --port 8001
# (or deploy it once to a free host and put the URL on Blackboard)
from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_iris

app = FastAPI(title="DATS 6401 Course Data Endpoint")

iris = load_iris(as_frame=True)
_names = list(iris.target_names)
_records = (iris.frame
            .assign(species=[_names[t] for t in iris.target])
            .drop(columns="target")
            .to_dict("records"))

@app.get("/")
def root():
    return {"endpoints": ["/data", "/data?species=setosa", "/summary", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/data")
def data(species: str | None = None):
    """All records, or filtered by ?species=. Unknown species -> 404 (on purpose:
    students practice handling non-200 responses)."""
    if species is None:
        return {"count": len(_records), "records": _records}
    sub = [r for r in _records if r["species"] == species]
    if not sub:
        raise HTTPException(status_code=404, detail=f"unknown species '{species}'")
    return {"count": len(sub), "records": sub}

@app.get("/summary")
def summary():
    out = {}
    for name in _names:
        vals = [r["sepal length (cm)"] for r in _records if r["species"] == name]
        out[name] = {"n": len(vals), "mean_sepal_length": round(sum(vals)/len(vals), 2)}
    return out
