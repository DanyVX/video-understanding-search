"""Local-only API with secure opaque IDs and bounded search responses."""

from __future__ import annotations

import re
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from vsearch.index.manifest import Manifest
from vsearch.index.transcripts import TranscriptIndex
from vsearch.settings import load_settings

VIDEO_ID = re.compile(r"^[a-f0-9]{24}$")
app = FastAPI(title="Video Understanding Search", version="0.1.0a0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=10, ge=1, le=50)


def manifest() -> Manifest:
    return Manifest(load_settings().data_dir)


def require_id(video_id: str) -> None:
    if not VIDEO_ID.fullmatch(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/videos")
def videos() -> list[dict[str, object]]:
    return manifest().list_videos()


@app.get("/v1/videos/{video_id}/moments")
def video_moments(video_id: str) -> list[dict[str, object]]:
    require_id(video_id)
    store = manifest()
    if store.get_video(video_id) is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return store.frames(video_id)


@app.post("/v1/search")
def search(request: SearchRequest) -> dict[str, object]:
    """Speech-only baseline until M2's visual model is integrated."""
    results = TranscriptIndex(load_settings().data_dir).search(request.query, request.limit)
    return {
        "query": request.query,
        "modality": "transcript_bm25",
        "results": [asdict(hit) for hit in results],
    }


@app.get("/v1/thumbnails/{video_id}/{frame_id}")
def thumbnail(video_id: str, frame_id: str) -> FileResponse:
    """Never accept caller-controlled filesystem paths."""
    require_id(video_id)
    require_id(frame_id)
    store = manifest()
    row = next((item for item in store.frames(video_id) if item["id"] == frame_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    root = store.root / "thumbs"
    target = (store.root / str(row["thumbnail"])).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail missing")
    return FileResponse(target, media_type="image/jpeg")


@app.delete("/v1/videos/{video_id}")
def delete(video_id: str) -> dict[str, bool]:
    from vsearch.ingest.pipeline import delete_video

    require_id(video_id)
    return {"deleted": delete_video(manifest(), video_id)}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    """Static UI with textContent instead of innerHTML for untrusted transcripts."""
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Moment Search</title>
<style>body{font:16px system-ui;margin:4rem auto;max-width:780px;padding:1rem;background:#111827;color:#e5e7eb}
input{padding:12px;width:70%;background:#1f2937;color:white;border:1px solid #4b5563}
button{padding:12px;cursor:pointer}article{padding:18px;margin-top:14px;background:#1f2937;border-radius:12px}
small{color:#9ca3af}</style></head><body><h1>Video Moment Search</h1>
<p>Timestamped transcript search (visual model integration pending)</p>
<form id="form"><input id="q" required maxlength="512" aria-label="Search videos" placeholder="Find spoken words…">
<button type="submit">Search</button></form><section id="results" aria-live="polite"></section>
<script>
const form=document.getElementById('form'), results=document.getElementById('results');
form.addEventListener('submit',async e=>{
e.preventDefault();results.replaceChildren();
const response=await fetch('/v1/search',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({query:document.getElementById('q').value})});
if(!response.ok){results.textContent='Search failed';return;}
const data=await response.json();
if(!data.results.length){results.textContent='No indexed transcript matches';return;}
for(const hit of data.results){
const card=document.createElement('article');
const title=document.createElement('strong');title.textContent='Video '+hit.video_id;
const time=document.createElement('small');time.textContent=' · '+hit.start.toFixed(2)+'s';
const snippet=document.createElement('p');snippet.textContent=hit.snippet;
card.append(title,time,snippet);results.append(card);
}});</script></body></html>"""
