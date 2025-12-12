# 🎬 AI 자막 제작기 (Subtitle Maker)

영상을 입력받아 오디오를 STT 모델로 텍스트로 변환하고, 변환된 텍스트에 타임스탬프를 부여하며, 사용자 친화적인 편집 에디터를 제공하여 수정 및 자막 파일(SRT/VTT/TXT) 다운로드가 가능한 AI 서비스입니다.

## ✨ 새로운 기능: 오디오 전처리

비디오 업로드 시 백그라운드에서 자동으로 오디오 전처리가 진행됩니다:
- **오디오 추출**: 비디오에서 고품질 오디오 분리
- **노이즈 제거**: 배경 소음 제거로 음성 명료도 향상
- **볼륨 정규화**: 일정한 음량 유지
- **샘플레이트 최적화**: Whisper 모델에 최적화된 16kHz 변환
- **실시간 진행상황 표시**: WebSocket을 통한 실시간 처리 상태 확인

이러한 전처리를 통해 **STT 정확도가 향상**되고 **처리 시간이 단축**됩니다.

## 🛠 기술 스택

- **Backend**: Python, FastAPI, OpenAI Whisper (STT), FFmpeg
- **Frontend**: React, WebSocket
- **오디오 처리**: ffmpeg-python, pydub, noisereduce, librosa
- **주요 기능**: 비디오 업로드, 오디오 전처리, 자동 자막 생성, 자막 편집, 다운로드

## 📁 프로젝트 구조

```
Subtitle-Maker/
├── subtitle_maker_backend/     # FastAPI 백엔드
│   ├── main.py                 # 메인 API 서버 (WebSocket 포함)
│   ├── audio_processor.py      # 오디오 전처리 모듈 (NEW)
│   ├── subtitle_utils.py       # 자막 파일 생성 유틸리티
│   └── requirements.txt        # Python 의존성
└── subtitle_maker_frontend/    # React 프론트엔드
    └── src/
        ├── App.js              # 메인 앱 컴포넌트 (WebSocket 연결)
        └── components/         # React 컴포넌트들
            ├── VideoUploader.js
            ├── VideoPlayer.js
            ├── SubtitleEditor.js
            ├── SubtitleDownloader.js
            └── PreprocessingStatus.js  # 전처리 상태 표시 (NEW)
```

## 🚀 시작하기

### 백엔드 설정

#### 사전 요구사항
- Python 3.8 이상
- **FFmpeg 설치 필요** (오디오/비디오 처리용)

FFmpeg 설치:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

#### 설치 및 실행

1. 백엔드 디렉토리로 이동:
```bash
cd subtitle_maker_backend
```

2. 가상환경 생성 및 활성화 (권장):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 서버 실행:
```bash
uvicorn main:app --reload
```

백엔드 서버는 `http://localhost:8000`에서 실행됩니다.

### 프론트엔드 설정

1. 프론트엔드 디렉토리로 이동:
```bash
cd subtitle_maker_frontend
```

2. 의존성 설치:
```bash
npm install
```

3. 개발 서버 실행:
```bash
npm start
```

프론트엔드 앱은 `http://localhost:3000`에서 실행됩니다.

## 📖 사용 방법

1. **비디오 업로드**: 메인 페이지에서 비디오 파일을 드래그 앤 드롭하거나 파일 선택 버튼을 클릭하여 업로드합니다.

2. **자동 전처리**: 업로드 완료 후 자동으로 오디오 전처리가 시작됩니다.
   - 실시간 진행상황 표시 (오디오 추출, 노이즈 제거, 볼륨 정규화)
   - 백그라운드에서 처리되므로 다른 작업 가능
   - 완료 시 자동 알림

3. **자막 생성**: 전처리 완료 후 "자막 생성하기" 버튼을 클릭합니다. 전처리된 고품질 오디오로 Whisper 모델이 텍스트 변환 및 타임스탬프를 부여합니다.

4. **자막 편집**: 생성된 자막을 편집 에디터에서 수정할 수 있습니다. 텍스트와 타임스탬프를 조정할 수 있습니다.

5. **다운로드**: 편집이 완료되면 SRT, VTT, TXT 형식으로 자막 파일을 다운로드할 수 있습니다.

## 🔌 API 엔드포인트

### 백엔드 API

- `POST /api/upload` - 비디오 파일 업로드 및 백그라운드 전처리 시작
- `WS /ws/{file_id}` - WebSocket 연결 (실시간 전처리 진행상황)
- `GET /api/preprocessing/{file_id}/status` - 전처리 상태 조회
- `POST /api/transcribe/{file_id}` - 비디오 STT 변환 (전처리된 오디오 사용)
- `GET /api/subtitles/{file_id}` - 자막 조회
- `POST /api/subtitles/{file_id}/edit` - 자막 편집
- `GET /api/subtitles/{file_id}/download/{format}` - 자막 다운로드 (SRT/VTT/TXT)
- `GET /api/video/{file_id}` - 비디오 파일 스트리밍

## 📝 주요 기능

### 기존 기능
- ✅ 비디오 파일 업로드 (드래그 앤 드롭 지원)
- ✅ OpenAI Whisper를 사용한 자동 STT 변환
- ✅ 타임스탬프 자동 부여
- ✅ 실시간 자막 미리보기
- ✅ 자막 텍스트 및 타임스탬프 편집
- ✅ 다중 포맷 지원 (SRT, VTT, TXT)
- ✅ 반응형 UI 디자인

### 새로운 전처리 기능
- ✨ 자동 오디오 추출 (FFmpeg)
- ✨ 배경 노이즈 제거 (noisereduce)
- ✨ 볼륨 정규화 (일정한 음량 유지)
- ✨ 샘플레이트 최적화 (16kHz - Whisper 최적화)
- ✨ WebSocket 기반 실시간 진행상황 표시
- ✨ 백그라운드 처리로 사용자 대기 시간 최소화
- ✨ 전처리된 고품질 오디오로 STT 정확도 향상

## ⚠️ 주의사항

- **FFmpeg 설치 필수**: 오디오 전처리 기능을 위해 시스템에 FFmpeg가 설치되어 있어야 합니다
- Whisper 모델은 처음 요청 시 자동으로 다운로드됩니다 (약 150MB)
- 큰 비디오 파일의 경우 전처리 및 변환에 시간이 걸릴 수 있습니다
- 업로드된 파일은 `subtitle_maker_backend/uploads/` 디렉토리에 저장됩니다
- 전처리된 오디오 파일은 `{file_id}_processed.wav` 형식으로 저장됩니다

## 🎯 성능 및 품질 개선

전처리 기능을 통한 개선 효과:
- **STT 정확도 향상**: 노이즈 제거로 음성 인식률 향상
- **처리 속도 개선**: 최적화된 오디오 포맷 사용
- **일관된 품질**: 볼륨 정규화로 안정적인 결과
- **사용자 경험**: 실시간 진행상황 표시로 투명한 처리 과정

## 📄 라이선스

이 프로젝트는 개인 사용 목적으로 개발되었습니다.

