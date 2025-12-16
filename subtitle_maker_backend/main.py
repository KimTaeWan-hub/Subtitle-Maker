from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import uuid
from pathlib import Path
import json
import asyncio
import logging

from subtitle_utils import create_srt, create_vtt, create_txt
from audio_preprocess import AudioPreprocessPipeline
from stt_models import get_global_model, set_global_model, STTModelFactory, ModelType

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Subtitle Maker API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드된 파일 저장 디렉토리
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# STT 모델은 stt_models.py에서 관리
# get_global_model()을 통해 전역 모델에 접근
# 필요시 set_global_model()로 모델 변경 가능

# 전처리 상태 관리 (메모리 기반)
preprocessing_status: Dict[str, Dict] = {}

# WebSocket 연결 관리
active_connections: Dict[str, WebSocket] = {}

# 오디오 전처리 파이프라인 인스턴스
audio_pipeline = AudioPreprocessPipeline(
    sample_rate=16000,
    separator_model="htdemucs",
    vad_threshold=0.5
)


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


@app.websocket("/ws/{file_id}")
async def websocket_endpoint(websocket: WebSocket, file_id: str):
    """WebSocket 연결 - 전처리 진행상황 실시간 전송"""
    await websocket.accept()
    active_connections[file_id] = websocket
    logger.info(f"WebSocket 연결됨: {file_id}")
    
    try:
        # 연결 즉시 현재 상태 전송
        current_status = preprocessing_status.get(file_id, {
            "status": "queued",
            "progress": 0,
            "message": "전처리 대기 중"
        })
        await websocket.send_json(current_status)
        logger.info(f"초기 상태 전송 [{file_id}]: {current_status}")
        
        # 연결 유지 및 메시지 수신 대기
        while True:
            data = await websocket.receive_text()
            logger.info(f"WebSocket 메시지 수신 [{file_id}]: {data}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket 연결 종료: {file_id}")
        if file_id in active_connections:
            del active_connections[file_id]
    except Exception as e:
        logger.error(f"WebSocket 오류: {str(e)}")
        if file_id in active_connections:
            del active_connections[file_id]


@app.get("/api/preprocessing/{file_id}/status")
async def get_preprocessing_status(file_id: str):
    """전처리 상태 조회 (HTTP 폴링 대안)"""
    try:
        status = preprocessing_status.get(file_id, {
            "status": "not_started",
            "progress": 0,
            "message": "전처리가 시작되지 않았습니다"
        })
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def send_progress_update(file_id: str, progress: int, message: str):
    """WebSocket을 통해 진행상황 전송"""
    # 상태 업데이트
    status = "processing" if progress < 100 else "completed"
    preprocessing_status[file_id] = {
        "status": status,
        "progress": progress,
        "message": message
    }
    
    # WebSocket으로 전송
    if file_id in active_connections:
        try:
            await active_connections[file_id].send_json({
                "status": status,
                "progress": progress,
                "message": message
            })
            logger.info(f"진행상황 전송 [{file_id}]: {progress}% - {message}")
        except Exception as e:
            logger.error(f"진행상황 전송 실패: {str(e)}")


async def preprocess_audio_task(file_id: str, video_path: Path):
    """백그라운드 오디오 전처리 작업"""
    try:
        # WebSocket 연결을 위한 짧은 대기 (최대 5초)
        logger.info(f"전처리 시작 대기 중: {file_id}")
        for _ in range(10):
            if file_id in active_connections:
                logger.info(f"WebSocket 연결 확인됨, 전처리 시작: {file_id}")
                break
            await asyncio.sleep(0.5)
        else:
            logger.warning(f"WebSocket 연결 없이 전처리 시작: {file_id}")
        
        # 전처리 상태 초기화
        preprocessing_status[file_id] = {
            "status": "processing",
            "progress": 0,
            "message": "전처리 시작"
        }
        
        # 오디오 추출 시작
        await send_progress_update(file_id, 10, "1단계: 오디오 추출 중...")
        
        # 음원 분리 시작
        await send_progress_update(file_id, 40, "2단계: 음원 분리 중...")
        
        # 음성 구간 탐지 시작
        await send_progress_update(file_id, 70, "3단계: 음성 구간 탐지 중...")
        
        # 출력 디렉토리 설정
        output_dir = UPLOAD_DIR / f"{file_id}_preprocessed"
        
        # 오디오 전처리 파이프라인 실행 (동기 함수를 별도 스레드에서 실행)
        def run_pipeline():
            return audio_pipeline.process(
                video_path=str(video_path),
                output_dir=str(output_dir),
                extract_vocals_only=True,
                detect_voice_segments=True,
                keep_intermediate_files=True  # 시각화를 위해 중간 파일 유지
            )
        
        # asyncio에서 동기 함수 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_pipeline)
        
        # 최종 오디오 파일을 표준 위치로 복사/이동
        final_audio = result.get('final_audio')
        if final_audio:
            import shutil
            standard_path = UPLOAD_DIR / f"{file_id}_processed.wav"
            shutil.copy(final_audio, standard_path)
            logger.info(f"최종 오디오 복사: {final_audio} -> {standard_path}")
        
        # 완료 상태 전송
        await send_progress_update(file_id, 100, "전처리 완료")
        
        logger.info(f"전처리 완료: {file_id} -> {final_audio}")
        
    except Exception as e:
        logger.error(f"전처리 실패 [{file_id}]: {str(e)}")
        preprocessing_status[file_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"전처리 실패: {str(e)}"
        }
        
        # 실패 알림 전송
        if file_id in active_connections:
            try:
                await active_connections[file_id].send_json({
                    "status": "failed",
                    "progress": 0,
                    "message": f"전처리 실패: {str(e)}"
                })
            except Exception:
                pass


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """비디오 파일 업로드 및 백그라운드 전처리 시작"""
    try:
        # 고유한 파일명 생성
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"파일 업로드 완료: {file_id} ({file.filename})")
        
        # 백그라운드에서 오디오 전처리 시작
        background_tasks.add_task(preprocess_audio_task, file_id, file_path)
        
        # 전처리 상태 초기화
        preprocessing_status[file_id] = {
            "status": "queued",
            "progress": 0,
            "message": "전처리 대기 중"
        }
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "preprocessing_status": "queued"
        }
    except Exception as e:
        logger.error(f"파일 업로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe/{file_id}")
async def transcribe_video(file_id: str):
    """비디오 파일을 STT로 변환하고 타임스탬프 생성"""
    try:
        # 전처리 상태 확인
        status = preprocessing_status.get(file_id, {})
        if status.get("status") == "processing":
            raise HTTPException(
                status_code=400,
                detail="전처리가 진행 중입니다. 잠시 후 다시 시도해주세요."
            )
        elif status.get("status") == "failed":
            raise HTTPException(
                status_code=400,
                detail="전처리에 실패했습니다. 파일을 다시 업로드해주세요."
            )
        
        # 전처리된 오디오 파일 찾기 (표준 위치)
        processed_audio_path = UPLOAD_DIR / f"{file_id}_processed.wav"
        
        # 전처리된 오디오가 없으면 원본 비디오 사용
        if processed_audio_path.exists():
            file_path = processed_audio_path
            logger.info(f"전처리된 오디오 사용: {file_path}")
        else:
            # 원본 비디오 파일 찾기
            file_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                candidate = UPLOAD_DIR / f"{file_id}{ext}"
                if candidate.exists():
                    file_path = candidate
                    break
            
            if not file_path:
                raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
            
            logger.warning(f"전처리된 오디오 없음, 원본 비디오 사용: {file_path}")
        
        # faster-whisper 모델 사용 (CTranslate2 기반)
        faster_model = STTModelFactory.create_model(
            model_type=ModelType.FASTER_WHISPER.value,
            model_size="medium",
            device="cpu",  # CPU 사용
            compute_type="int8"  # CPU에서는 int8 사용 (float16은 GPU만 지원)
        )
        set_global_model(faster_model)

        # # Whisper large 모델 사용
        # large_model = STTModelFactory.create_model(
        #     model_type=ModelType.WHISPER.value,
        #     model_size="large",
        #     device="cuda"
        # )
        # set_global_model(large_model)
        
        # STT 모델로 처리
        logger.info(f"STT 처리 시작: {file_id}")
        model = get_global_model()
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
        
        logger.info(f"STT 처리 완료: {file_id} ({len(segments)} 세그먼트)")
        
        return {
            "file_id": file_id,
            "segments": segments
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT 처리 실패: {str(e)}")
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


@app.get("/api/verify/{file_id}")
async def verify_preprocessing(file_id: str):
    """전처리 검증 결과 반환 (수치 데이터만)"""
    try:
        import soundfile as sf
        import numpy as np
        
        processed_path = UPLOAD_DIR / f"{file_id}_processed.wav"
        
        if not processed_path.exists():
            raise HTTPException(status_code=404, detail="전처리된 파일을 찾을 수 없습니다")
        
        # 오디오 로드 및 분석
        audio, sr = sf.read(str(processed_path))
        
        rms_level = float(np.sqrt(np.mean(audio**2)))
        max_amp = float(np.abs(audio).max())
        clipping_count = int(np.sum(np.abs(audio) >= 0.99))
        clipping_ratio = float(clipping_count / len(audio))
        
        # 검증 항목
        checks = {
            "correct_sample_rate": sr == 16000,
            "mono_channel": audio.ndim == 1,
            "proper_volume": 0.05 <= rms_level <= 0.15,
            "no_clipping": clipping_ratio < 0.01,
            "not_silent": max_amp > 0.001,
        }
        
        all_passed = all(checks.values())
        
        return {
            "file_id": file_id,
            "status": "passed" if all_passed else "failed",
            "audio_info": {
                "sample_rate": int(sr),
                "duration_seconds": float(len(audio) / sr),
                "channels": 1,
                "total_samples": int(len(audio)),
            },
            "metrics": {
                "rms_level": round(rms_level, 4),
                "max_amplitude": round(max_amp, 4),
                "clipping_count": clipping_count,
                "clipping_ratio": round(clipping_ratio * 100, 4),  # 퍼센트
                "std_deviation": round(float(np.std(audio)), 4),
            },
            "checks": checks,
            "recommendation": "전처리가 성공적으로 완료되었습니다." if all_passed else "일부 검증 항목에서 문제가 발견되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"검증 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"검증 중 오류 발생: {str(e)}")


@app.get("/api/waveform/{file_id}")
async def get_waveform_visualization(file_id: str):
    """전처리 단계별 파형 시각화 이미지 반환"""
    try:
        # 전처리 디렉토리에서 시각화 이미지 찾기
        preprocessed_dir = UPLOAD_DIR / f"{file_id}_preprocessed"
        
        # 가능한 파형 이미지 파일 찾기
        waveform_path = None
        for pattern in ["*_waveform.png", "waveform.png"]:
            import glob
            matches = glob.glob(str(preprocessed_dir / pattern))
            if matches:
                waveform_path = Path(matches[0])
                break
        
        if not waveform_path or not waveform_path.exists():
            raise HTTPException(status_code=404, detail="파형 시각화 이미지를 찾을 수 없습니다")
        
        return FileResponse(
            path=str(waveform_path),
            media_type="image/png",
            filename=f"{file_id}_waveform.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파형 이미지 반환 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"파형 이미지 반환 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)