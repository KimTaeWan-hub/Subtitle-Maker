import React, { useState } from 'react';
import './App.css';
import VideoUploader from './components/VideoUploader';
import VideoPlayer from './components/VideoPlayer';
import SubtitleEditor from './components/SubtitleEditor';
import SubtitleDownloader from './components/SubtitleDownloader';

function App() {
  const [fileId, setFileId] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [segments, setSegments] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleUploadSuccess = (uploadedFileId, filename) => {
    setFileId(uploadedFileId);
    setVideoUrl(`http://localhost:8000/api/video/${uploadedFileId}`);
    setSegments([]); // 새 파일 업로드 시 자막 초기화
  };

  const handleTranscribeSuccess = (transcribedSegments) => {
    setSegments(transcribedSegments);
    setIsProcessing(false);
  };

  const handleSubtitleUpdate = (updatedSegments) => {
    setSegments(updatedSegments);
  };

  const handleTranscribe = async () => {
    if (!fileId) {
      alert('먼저 비디오를 업로드해주세요.');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch(`http://localhost:8000/api/transcribe/${fileId}`, {
        method: 'POST',
      });
      const data = await response.json();
      handleTranscribeSuccess(data.segments);
    } catch (error) {
      console.error('변환 실패:', error);
      alert('변환 중 오류가 발생했습니다.');
      setIsProcessing(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Subtitle Maker</h1>
        <p>영상을 업로드하고 자동으로 자막을 생성하세요</p>
      </header>

      <main className="App-main">
        <div className="main-layout">
          {/* 좌측: 영상 영역 */}
          <div className="left-panel">
            <div className="video-section">
              {!videoUrl ? (
                <VideoUploader onUploadSuccess={handleUploadSuccess} />
              ) : (
                <>
                  <VideoPlayer videoUrl={videoUrl} segments={segments} />
                  <div className="video-actions">
                    <button
                      className="btn btn-secondary"
                      onClick={() => {
                        setFileId(null);
                        setVideoUrl(null);
                        setSegments([]);
                      }}
                    >
                      새 영상 업로드
                    </button>
                  </div>
                </>
              )}
            </div>
            {segments.length > 0 && (
              <div className="download-section-left">
                <SubtitleDownloader fileId={fileId} />
              </div>
            )}
          </div>

          {/* 우측: 자막 영역 */}
          <div className="right-panel">
            <div className="subtitle-section">
              {segments.length === 0 ? (
                <div className="subtitle-empty-state">
                  <h2>자막 생성</h2>
                  <p>좌측에서 영상을 업로드한 후, 아래 버튼을 클릭하여 자막을 생성하세요.</p>
                  <button
                    className="btn btn-primary"
                    onClick={handleTranscribe}
                    disabled={!fileId || isProcessing}
                  >
                    {isProcessing ? '변환 중...' : '자막 생성하기'}
                  </button>
                </div>
              ) : (
                <SubtitleEditor
                  segments={segments}
                  fileId={fileId}
                  onUpdate={handleSubtitleUpdate}
                  onRegenerate={handleTranscribe}
                  isProcessing={isProcessing}
                />
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
