"""Local-only API with secure opaque IDs and bounded search responses."""

from __future__ import annotations

import re
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from vsearch.index.manifest import Manifest
from vsearch.retrieve.service import search_moments
from vsearch.settings import load_settings

VIDEO_ID = re.compile(r"^[a-f0-9]{24}$")
app = FastAPI(title="Video Understanding Search", version="0.1.0a0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=10, ge=1, le=50)
    include_visual: bool = False


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
    """Text-first hybrid retrieval; visual inference explicitly optional."""
    try:
        results = search_moments(
            load_settings().data_dir,
            request.query,
            limit=request.limit,
            include_visual=request.include_visual,
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=type(exc).__name__) from None
    return {
        "modality": "hybrid" if request.include_visual else "transcript_bm25",
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


@app.get("/v1/media/{video_id}")
def media(video_id: str) -> FileResponse:
    """Serve only explicitly retained local sources; allows HTTP Range seeking."""
    from pathlib import Path

    require_id(video_id)
    record = manifest().get_video(video_id)
    if record is None or not record["source_path"]:
        raise HTTPException(status_code=404, detail="Media not retained")
    candidate = Path(str(record["source_path"]))
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media source unavailable")
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.delete("/v1/videos/{video_id}")
def delete(video_id: str) -> dict[str, bool]:
    from vsearch.ingest.pipeline import delete_video

    require_id(video_id)
    removed = delete_video(manifest(), video_id)
    from vsearch.index.vectors import SQLiteVectorStore

    SQLiteVectorStore(load_settings().data_dir).delete_video(video_id)
    return {"deleted": removed}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    """Static UI with textContent instead of innerHTML for untrusted transcripts."""
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Moment Search</title>
<style>body{font:16px system-ui;margin:4rem auto;max-width:780px;padding:1rem;
background:#111827;color:#e5e7eb}
input{padding:12px;width:70%;background:#1f2937;color:white;border:1px solid #4b5563}
button{padding:12px;cursor:pointer}article{padding:18px;margin-top:14px;background:#1f2937;border-radius:12px}
small{color:#9ca3af}</style></head><body><h1>Video Moment Search</h1>
<p>Timestamped transcript search (visual model integration pending)</p>
<form id="form"><input id="q" required maxlength="512" aria-label="Search videos" 
placeholder="Find spoken words…">
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
const video=document.createElement('video');video.controls=true;
video.preload='metadata';video.style.width='100%';
video.src='/v1/media/'+encodeURIComponent(hit.video_id);
video.addEventListener('loadedmetadata',()=>{video.currentTime=hit.start;},{once:true});
card.append(title,time,snippet,video);results.append(card);
}});</script></body></html>"""
