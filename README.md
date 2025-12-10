# 🎬 AI 자막 제작기 (Subtitle Maker)

영상을 입력받아 오디오를 STT 모델로 텍스트로 변환하고, 변환된 텍스트에 타임스탬프를 부여하며, 사용자 친화적인 편집 에디터를 제공하여 수정 및 자막 파일(SRT/VTT/TXT) 다운로드가 가능한 AI 서비스입니다.

## 🛠 기술 스택

- **Backend**: Python, FastAPI, OpenAI Whisper (STT)
- **Frontend**: React
- **주요 기능**: 비디오 업로드, 자동 자막 생성, 자막 편집, 다운로드

## 📁 프로젝트 구조

```
Subtitle-Maker/
├── subtitle_maker_backend/     # FastAPI 백엔드
│   ├── main.py                 # 메인 API 서버
│   ├── subtitle_utils.py       # 자막 파일 생성 유틸리티
│   └── requirements.txt        # Python 의존성
└── subtitle_maker_frontend/    # React 프론트엔드
    └── src/
        ├── App.js              # 메인 앱 컴포넌트
        └── components/         # React 컴포넌트들
            ├── VideoUploader.js
            ├── VideoPlayer.js
            ├── SubtitleEditor.js
            └── SubtitleDownloader.js
```

## 🚀 시작하기

### 백엔드 설정

1. 백엔드 디렉토리로 이동:
```bash
cd subtitle_maker_backend
```

2. 가상환경 생성 및 활성화 (선택사항):
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

2. **자막 생성**: 업로드된 비디오를 확인한 후 "자막 생성하기" 버튼을 클릭합니다. Whisper 모델이 오디오를 텍스트로 변환하고 타임스탬프를 부여합니다.

3. **자막 편집**: 생성된 자막을 편집 에디터에서 수정할 수 있습니다. 텍스트와 타임스탬프를 조정할 수 있습니다.

4. **다운로드**: 편집이 완료되면 SRT, VTT, TXT 형식으로 자막 파일을 다운로드할 수 있습니다.

## 🔌 API 엔드포인트

### 백엔드 API

- `POST /api/upload` - 비디오 파일 업로드
- `POST /api/transcribe/{file_id}` - 비디오 STT 변환
- `GET /api/subtitles/{file_id}` - 자막 조회
- `POST /api/subtitles/{file_id}/edit` - 자막 편집
- `GET /api/subtitles/{file_id}/download/{format}` - 자막 다운로드 (SRT/VTT/TXT)
- `GET /api/video/{file_id}` - 비디오 파일 스트리밍

## 📝 주요 기능

- ✅ 비디오 파일 업로드 (드래그 앤 드롭 지원)
- ✅ OpenAI Whisper를 사용한 자동 STT 변환
- ✅ 타임스탬프 자동 부여
- ✅ 실시간 자막 미리보기
- ✅ 자막 텍스트 및 타임스탬프 편집
- ✅ 다중 포맷 지원 (SRT, VTT, TXT)
- ✅ 반응형 UI 디자인

## ⚠️ 주의사항

- Whisper 모델은 처음 요청 시 자동으로 다운로드됩니다 (약 150MB)
- 큰 비디오 파일의 경우 변환에 시간이 걸릴 수 있습니다
- 업로드된 파일은 `subtitle_maker_backend/uploads/` 디렉토리에 저장됩니다

## 📄 라이선스

이 프로젝트는 개인 사용 목적으로 개발되었습니다.

