# 🎬 AI 자막 제작기 (Subtitle Maker)

영상을 입력받아 오디오를 STT 모델로 텍스트로 변환하고, 변환된 텍스트에 타임스탬프를 부여하며, 사용자 친화적인 편집 에디터를 제공하여 수정 및 자막 파일(SRT/VTT/TXT) 다운로드가 가능한 AI 서비스입니다.

## ✨ 주요 기능: 오디오 전처리 파이프라인

비디오 업로드 시 백그라운드에서 자동으로 3단계 오디오 전처리가 진행됩니다:
- **1단계 - 오디오 추출**: 비디오에서 오디오 추출 및 16kHz 모노 변환 (FFmpeg)
- **2단계 - 음원 분리**: Demucs 모델(htdemucs)을 사용한 보컬 추출
- **3단계 - 음성 구간 탐지**: Silero VAD를 사용한 음성 구간 자동 탐지
- **실시간 진행상황 표시**: WebSocket을 통한 실시간 처리 상태 확인

이러한 전처리를 통해 **STT 정확도가 향상**되고 **처리 시간이 단축**됩니다.

## 🛠 기술 스택

- **Backend**: Python, FastAPI, Faster-Whisper (STT), FFmpeg
- **Frontend**: React, WebSocket
- **오디오 처리**: ffmpeg-python, Demucs, Silero VAD, librosa, soundfile
- **STT 모델**: Faster-Whisper (CTranslate2 기반, medium 모델)
- **주요 기능**: 비디오 업로드, 오디오 전처리, 자동 자막 생성, 자막 편집, 다운로드

## 📁 프로젝트 구조

```
Subtitle-Maker/
├── subtitle_maker_backend/          # FastAPI 백엔드
│   ├── main.py                      # 메인 API 서버 (WebSocket 포함)
│   ├── audio_preprocess.py          # 오디오 전처리 파이프라인
│   ├── audio_extractor.py           # 비디오에서 오디오 추출
│   ├── audio_separator.py           # Demucs 음원 분리
│   ├── audio_detector.py            # Silero VAD 음성 구간 탐지
│   ├── stt_models.py                # STT 모델 관리 (Whisper, Faster-Whisper)
│   ├── subtitle_utils.py            # 자막 파일 생성 유틸리티 (SRT/VTT/TXT)
│   ├── verify_preprocessing.py      # 전처리 검증 스크립트
│   ├── requirements_faster_whisper.txt  # Python 의존성
│   └── uploads/                     # 업로드된 파일 저장 디렉토리
└── subtitle_maker_frontend/         # React 프론트엔드
    └── src/
        ├── App.js                   # 메인 앱 컴포넌트 (WebSocket 연결)
        └── components/              # React 컴포넌트들
            ├── VideoUploader.js     # 비디오 업로드
            ├── VideoPlayer.js       # 비디오 플레이어
            ├── SubtitleEditor.js    # 자막 편집기
            ├── SubtitleDownloader.js  # 자막 다운로드
            └── PreprocessingStatus.js  # 전처리 상태 표시
```

## 🚀 시작하기

### 백엔드 설정

#### 사전 요구사항
- Python 3.8 이상 (3.10 권장)
- **FFmpeg 설치 필수** (오디오/비디오 처리용)

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
pip install -r requirements_faster_whisper.txt
```

**참고**: PyTorch CPU 버전 설치가 필요한 경우:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Demucs 최신 버전 설치:
```bash
pip install "git+https://github.com/facebookresearch/demucs#egg=demucs"
```

4. 서버 실행:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
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

2. **자동 전처리**: 업로드 완료 후 자동으로 3단계 오디오 전처리가 시작됩니다.
   - 1단계: 오디오 추출 (10% 진행률)
   - 2단계: 음원 분리 (40% 진행률)
   - 3단계: 음성 구간 탐지 (70% 진행률)
   - 실시간 진행상황 표시 (WebSocket)
   - 백그라운드에서 처리되므로 다른 작업 가능
   - 완료 시 자동 알림 (100%)

3. **자막 생성**: 전처리 완료 후 "자막 생성하기" 버튼을 클릭합니다. 
   - Faster-Whisper 모델(medium)이 전처리된 오디오를 텍스트로 변환
   - 단어 단위 타임스탬프 자동 부여
   - VAD 필터 적용으로 정확도 향상

4. **자막 편집**: 생성된 자막을 편집 에디터에서 수정할 수 있습니다. 
   - 텍스트 내용 수정
   - 타임스탬프 조정
   - 세그먼트 클릭 시 해당 시점으로 비디오 이동

5. **다운로드**: 편집이 완료되면 SRT, VTT, TXT 형식으로 자막 파일을 다운로드할 수 있습니다.

## 🔌 API 엔드포인트

### 백엔드 API

- `POST /api/upload` - 비디오 파일 업로드 및 백그라운드 전처리 시작
- `WS /ws/{file_id}` - WebSocket 연결 (실시간 전처리 진행상황)
- `GET /api/preprocessing/{file_id}/status` - 전처리 상태 조회 (HTTP 폴링 대안)
- `POST /api/transcribe/{file_id}` - 비디오 STT 변환 (전처리된 오디오 사용)
- `GET /api/subtitles/{file_id}` - 자막 조회
- `POST /api/subtitles/{file_id}/edit` - 자막 편집
- `GET /api/subtitles/{file_id}/download/{format}` - 자막 다운로드 (SRT/VTT/TXT)
- `GET /api/video/{file_id}` - 비디오 파일 스트리밍
- `GET /api/verify/{file_id}` - 전처리 검증 결과 조회 (오디오 품질 메트릭)
- `GET /api/waveform/{file_id}` - 전처리 단계별 파형 시각화 이미지

## 📝 주요 기능

### 핵심 기능
- ✅ 비디오 파일 업로드 (드래그 앤 드롭 지원)
- ✅ Faster-Whisper를 사용한 고속 STT 변환 (CTranslate2 기반)
- ✅ 단어 단위 타임스탬프 자동 부여
- ✅ 실시간 자막 미리보기 및 비디오 동기화
- ✅ 자막 텍스트 및 타임스탬프 편집
- ✅ 다중 포맷 지원 (SRT, VTT, TXT)
- ✅ 반응형 UI 디자인

### 오디오 전처리 파이프라인
- ✨ **1단계: 오디오 추출** - FFmpeg를 사용한 비디오에서 오디오 추출 및 16kHz 모노 변환
- ✨ **2단계: 음원 분리** - Demucs(htdemucs) 모델을 사용한 보컬/악기 분리
- ✨ **3단계: 음성 구간 탐지** - Silero VAD를 사용한 음성 구간 자동 탐지
- ✨ WebSocket 기반 실시간 진행상황 표시 (10% → 40% → 70% → 100%)
- ✨ 백그라운드 처리로 사용자 대기 시간 최소화
- ✨ 전처리 검증 및 파형 시각화 지원
- ✨ 전처리된 고품질 오디오로 STT 정확도 향상

## ⚠️ 주의사항

- **FFmpeg 설치 필수**: 오디오 전처리 기능을 위해 시스템에 FFmpeg가 설치되어 있어야 합니다
- **Faster-Whisper 모델**: 처음 요청 시 자동으로 다운로드됩니다 (medium 모델 약 1.5GB)
- **Demucs 모델**: 처음 사용 시 htdemucs 모델이 자동으로 다운로드됩니다 (약 2GB)
- **처리 시간**: 큰 비디오 파일의 경우 전처리(특히 음원 분리) 및 STT 변환에 시간이 걸릴 수 있습니다
- **저장 위치**: 
  - 업로드된 파일: `subtitle_maker_backend/uploads/{file_id}.{ext}`
  - 전처리 결과: `subtitle_maker_backend/uploads/{file_id}_preprocessed/`
  - 최종 오디오: `subtitle_maker_backend/uploads/{file_id}_processed.wav`
  - 자막 데이터: `subtitle_maker_backend/uploads/{file_id}_transcription.json`

## 🎯 성능 및 품질 개선

전처리 파이프라인을 통한 개선 효과:
- **STT 정확도 향상**: Demucs 음원 분리로 보컬만 추출하여 음성 인식률 향상
- **처리 효율 개선**: VAD로 음성 구간만 탐지하여 불필요한 처리 최소화
- **최적화된 포맷**: 16kHz 모노 오디오로 Whisper 모델에 최적화
- **사용자 경험**: WebSocket 기반 실시간 진행상황 표시로 투명한 처리 과정
- **품질 검증**: 전처리 결과 검증 API로 오디오 품질 확인 가능

## 📄 라이선스

이 프로젝트는 개인 사용 목적으로 개발되었습니다.

