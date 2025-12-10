from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import whisper
from pathlib import Path
import json
from dotenv import load_dotenv

from subtitle_utils import create_srt, create_vtt, create_txt

load_dotenv()

app = FastAPI(title="Subtitle Maker API")

HOST_URL=os.getenv("BACKEND_URL")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[HOST_URL],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드된 파일 저장 디렉토리
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# STT 모델 로드 (처음 요청 시 로드)
model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("small")  # base 모델 사용 (더 빠름) # tiny, base, small, medium, large
    return model


class SubtitleSegment(BaseModel):
    id: int
    start_time: float
    end_time: float
    text: str


class SubtitleEditRequest(BaseModel):
    segments: List[SubtitleSegment]


@app.get("/")
async def root():
    return {"message": "Subtitle Maker API"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """비디오 파일 업로드"""
    try:
        # 고유한 파일명 생성
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe/{file_id}")
async def transcribe_video(file_id: str):
    """비디오 파일을 STT로 변환하고 타임스탬프 생성"""
    try:
        # 파일 찾기
        file_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            candidate = UPLOAD_DIR / f"{file_id}{ext}"
            if candidate.exists():
                file_path = candidate
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
        
        # Whisper 모델로 STT 처리
        model = get_model()
        result = model.transcribe(str(file_path), word_timestamps=True)
        
        # 세그먼트 생성
        segments = []
        for idx, segment in enumerate(result["segments"], 1):
            segments.append({
                "id": idx,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"].strip()
            })
        
        # 결과 저장
        result_path = UPLOAD_DIR / f"{file_id}_transcription.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        return {
            "file_id": file_id,
            "segments": segments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/subtitles/{file_id}/edit")
async def edit_subtitles(file_id: str, request: SubtitleEditRequest):
    """자막 편집"""
    try:
        # 편집된 세그먼트 저장
        result_path = UPLOAD_DIR / f"{file_id}_transcription.json"
        segments_data = [seg.dict() for seg in request.segments]
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)
        
        return {
            "file_id": file_id,
            "message": "자막이 성공적으로 저장되었습니다",
            "segments": segments_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subtitles/{file_id}")
async def get_subtitles(file_id: str):
    """저장된 자막 조회"""
    try:
        result_path = UPLOAD_DIR / f"{file_id}_transcription.json"
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="자막을 찾을 수 없습니다")
        
        with open(result_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        
        return {
            "file_id": file_id,
            "segments": segments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subtitles/{file_id}/download/{format}")
async def download_subtitle(file_id: str, format: str):
    """자막 파일 다운로드 (SRT/VTT/TXT)"""
    try:
        result_path = UPLOAD_DIR / f"{file_id}_transcription.json"
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="자막을 찾을 수 없습니다")
        
        with open(result_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        
        # 포맷에 따라 파일 생성
        if format.lower() == "srt":
            subtitle_content = create_srt(segments)
            filename = f"{file_id}.srt"
            content_type = "text/plain"
        elif format.lower() == "vtt":
            subtitle_content = create_vtt(segments)
            filename = f"{file_id}.vtt"
            content_type = "text/vtt"
        elif format.lower() == "txt":
            subtitle_content = create_txt(segments)
            filename = f"{file_id}.txt"
            content_type = "text/plain"
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 포맷입니다 (SRT/VTT/TXT만 지원)")
        
        # 임시 파일로 저장
        output_path = UPLOAD_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(subtitle_content)
        
        return FileResponse(
            path=str(output_path),
            filename=filename,
            media_type=content_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/{file_id}")
async def get_video(file_id: str):
    """업로드된 비디오 파일 스트리밍"""
    try:
        file_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            candidate = UPLOAD_DIR / f"{file_id}{ext}"
            if candidate.exists():
                file_path = candidate
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다")
        
        return FileResponse(
            path=str(file_path),
            media_type="video/mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)