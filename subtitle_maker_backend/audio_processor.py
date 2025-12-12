"""
오디오 전처리 모듈
비디오에서 오디오 추출, 노이즈 제거, 볼륨 정규화, 샘플레이트 최적화
"""

import os
import logging
from pathlib import Path
from typing import Callable, Optional
import ffmpeg
import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa
from pydub import AudioSegment
from pydub.effects import normalize

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """오디오 전처리를 담당하는 클래스"""
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Args:
            target_sample_rate: 목표 샘플 레이트 (기본값: 16000 - Whisper 최적)
        """
        self.target_sample_rate = target_sample_rate
    
    async def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Path:
        """
        비디오 파일을 전처리하여 최적화된 오디오 파일 생성
        
        Args:
            video_path: 입력 비디오 파일 경로
            output_dir: 출력 디렉토리
            progress_callback: 진행상황 콜백 함수 (progress: int, message: str)
        
        Returns:
            처리된 오디오 파일 경로
        """
        try:
            file_id = video_path.stem
            temp_audio_path = output_dir / f"{file_id}_temp.wav"
            final_audio_path = output_dir / f"{file_id}_processed.wav"
            
            # 1단계: 오디오 추출 (0-20%)
            if progress_callback:
                await progress_callback(5, "오디오 추출 시작")
            
            audio_data, sr = self._extract_audio(video_path, temp_audio_path)
            
            if progress_callback:
                await progress_callback(20, "오디오 추출 완료")
            
            # 2단계: 샘플레이트 최적화 (20-35%)
            if progress_callback:
                await progress_callback(25, "샘플레이트 최적화 중")
            
            if sr != self.target_sample_rate:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sr,
                    target_sr=self.target_sample_rate
                )
                sr = self.target_sample_rate
            
            if progress_callback:
                await progress_callback(35, "샘플레이트 최적화 완료")
            
            # 3단계: 노이즈 제거 (35-70%)
            if progress_callback:
                await progress_callback(40, "노이즈 제거 시작")
            
            audio_data = self._reduce_noise(audio_data, sr)
            
            if progress_callback:
                await progress_callback(70, "노이즈 제거 완료")
            
            # 4단계: 볼륨 정규화 (70-90%)
            if progress_callback:
                await progress_callback(75, "볼륨 정규화 시작")
            
            audio_data = self._normalize_volume(audio_data)
            
            if progress_callback:
                await progress_callback(90, "볼륨 정규화 완료")
            
            # 5단계: 최종 파일 저장 (90-100%)
            if progress_callback:
                await progress_callback(95, "파일 저장 중")
            
            # 오디오 데이터 저장
            sf.write(str(final_audio_path), audio_data, sr)
            
            # 임시 파일 삭제
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            
            if progress_callback:
                await progress_callback(100, "전처리 완료")
            
            logger.info(f"오디오 전처리 완료: {final_audio_path}")
            return final_audio_path
            
        except Exception as e:
            logger.error(f"오디오 전처리 실패: {str(e)}")
            raise
    
    def _extract_audio(self, video_path: Path, output_path: Path) -> tuple[np.ndarray, int]:
        """
        비디오에서 오디오 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 임시 오디오 파일 경로
        
        Returns:
            (audio_data, sample_rate) 튜플
        """
        try:
            # ffmpeg로 오디오 추출 (16bit PCM WAV)
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le',
                ac=1,  # 모노로 변환
                ar=self.target_sample_rate,
                loglevel='error'
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # 추출된 오디오 로드
            audio_data, sr = sf.read(str(output_path))
            
            logger.info(f"오디오 추출 완료: {video_path.name} -> {output_path.name}")
            return audio_data, sr
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg 오류: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except Exception as e:
            logger.error(f"오디오 추출 실패: {str(e)}")
            raise
    
    def _reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        노이즈 제거
        
        Args:
            audio_data: 오디오 데이터
            sample_rate: 샘플 레이트
        
        Returns:
            노이즈가 제거된 오디오 데이터
        """
        try:
            # noisereduce를 사용한 노이즈 제거
            # stationary=True: 지속적인 배경 노이즈 제거
            # prop_decrease: 노이즈 감소 비율 (0.0~1.0)
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            logger.info("노이즈 제거 완료")
            return reduced_noise
            
        except Exception as e:
            logger.warning(f"노이즈 제거 실패, 원본 사용: {str(e)}")
            return audio_data
    
    def _normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """
        볼륨 정규화
        
        Args:
            audio_data: 오디오 데이터
        
        Returns:
            정규화된 오디오 데이터
        """
        try:
            # RMS 정규화
            # 목표 RMS 레벨 설정 (약 -20 dBFS)
            target_rms = 0.1
            
            # 현재 RMS 계산
            current_rms = np.sqrt(np.mean(audio_data**2))
            
            if current_rms > 0:
                # 정규화 계수 계산
                normalization_factor = target_rms / current_rms
                normalized_audio = audio_data * normalization_factor
                
                # 클리핑 방지 (-1.0 ~ 1.0 범위)
                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                
                logger.info(f"볼륨 정규화 완료: {current_rms:.4f} -> {target_rms:.4f}")
                return normalized_audio
            else:
                logger.warning("오디오 신호가 너무 약함, 정규화 건너뜀")
                return audio_data
                
        except Exception as e:
            logger.warning(f"볼륨 정규화 실패, 원본 사용: {str(e)}")
            return audio_data
    
    def get_processed_audio_path(self, file_id: str, output_dir: Path) -> Optional[Path]:
        """
        전처리된 오디오 파일 경로 반환
        
        Args:
            file_id: 파일 ID
            output_dir: 출력 디렉토리
        
        Returns:
            전처리된 오디오 파일 경로 (존재하지 않으면 None)
        """
        processed_path = output_dir / f"{file_id}_processed.wav"
        return processed_path if processed_path.exists() else None

