"""
STT 모델 관리 모듈
- Whisper 모델과 Faster-Whisper 모델을 관리
- 모델 로드 및 음성 인식 처리
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """지원하는 STT 모델 타입"""
    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"


class WhisperModelManager:
    """OpenAI Whisper 모델 관리 클래스"""
    
    def __init__(self, model_size: str = "medium", device: Optional[str] = None):
        """
        Whisper 모델 초기화
        
        Args:
            model_size: 모델 크기 (tiny, base, small, medium, large, turbo)
            device: 실행 디바이스 (cuda, cpu). None이면 자동 선택
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._model_type = ModelType.WHISPER
        
    def load_model(self) -> None:
        """모델을 메모리에 로드"""
        if self.model is None:
            try:
                import whisper
                
                logger.info(f"Whisper 모델 로드 중: {self.model_size}")
                
                if self.device:
                    self.model = whisper.load_model(self.model_size, device=self.device)
                else:
                    self.model = whisper.load_model(self.model_size)
                
                logger.info(f"Whisper 모델 로드 완료: {self.model_size}")
                logger.info(
                    f"모델 정보 - "
                    f"Multilingual: {self.model.is_multilingual}, "
                    f"Device: {self.model.device}"
                )
                
            except Exception as e:
                logger.error(f"Whisper 모델 로드 실패: {str(e)}")
                raise
    
    def get_model(self):
        """모델 인스턴스 반환 (lazy loading)"""
        if self.model is None:
            self.load_model()
        return self.model
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        word_timestamps: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        음성을 텍스트로 변환
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (None이면 자동 감지)
            word_timestamps: 단어 단위 타임스탬프 생성 여부
            **kwargs: 추가 transcribe 옵션
            
        Returns:
            transcription 결과 딕셔너리
        """
        model = self.get_model()
        
        logger.info(f"Whisper transcription 시작: {audio_path}")
        
        try:
            # 기본 옵션 설정
            transcribe_options = {
                "word_timestamps": word_timestamps,
            }
            
            # 언어 지정
            if language:
                transcribe_options["language"] = language
            
            # 추가 옵션 병합
            transcribe_options.update(kwargs)
            
            # 음성 인식 수행
            result = model.transcribe(str(audio_path), **transcribe_options)
            
            logger.info(
                f"Whisper transcription 완료: "
                f"{len(result.get('segments', []))} 세그먼트, "
                f"언어: {result.get('language', 'unknown')}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription 실패: {str(e)}")
            raise
    
    @property
    def model_type(self) -> str:
        """모델 타입 반환"""
        return self._model_type.value


class FasterWhisperModelManager:
    """Faster-Whisper 모델 관리 클래스"""
    
    def __init__(
        self, 
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "default"
    ):
        """
        Faster-Whisper 모델 초기화
        
        Args:
            model_size: 모델 크기 (tiny, base, small, medium, large-v3, turbo)
            device: 실행 디바이스 (cuda, cpu)
            compute_type: 연산 타입 (default, float16, int8, int8_float16)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._model_type = ModelType.FASTER_WHISPER
        
    def load_model(self) -> None:
        """모델을 메모리에 로드"""
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                
                logger.info(
                    f"Faster-Whisper 모델 로드 중: {self.model_size}, "
                    f"device={self.device}, compute_type={self.compute_type}"
                )
                
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                
                logger.info(f"Faster-Whisper 모델 로드 완료: {self.model_size}")
                
            except Exception as e:
                logger.error(f"Faster-Whisper 모델 로드 실패: {str(e)}")
                raise
    
    def get_model(self):
        """모델 인스턴스 반환 (lazy loading)"""
        if self.model is None:
            self.load_model()
        return self.model
    
    def transcribe(
        self, 
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        음성을 텍스트로 변환
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (None이면 자동 감지)
            beam_size: 빔 서치 크기 (1=greedy, 5=default)
            word_timestamps: 단어 단위 타임스탬프 생성 여부
            **kwargs: 추가 transcribe 옵션
            
        Returns:
            transcription 결과 딕셔너리 (Whisper 형식과 호환)
        """
        model = self.get_model()
        
        logger.info(f"Faster-Whisper transcription 시작: {audio_path}")
        
        try:
            # 기본 옵션 설정
            transcribe_options = {
                "beam_size": beam_size,
                "word_timestamps": word_timestamps,
            }
            
            # 언어 지정
            if language:
                transcribe_options["language"] = language
            
            # 추가 옵션 병합
            transcribe_options.update(kwargs)
            
            # 음성 인식 수행
            segments, info = model.transcribe(str(audio_path), **transcribe_options)
            
            # Whisper 형식과 호환되도록 결과 변환
            result_segments = []
            for segment in segments:
                result_segments.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                })
            
            result = {
                "text": " ".join([seg["text"] for seg in result_segments]),
                "segments": result_segments,
                "language": info.language,
                "duration": info.duration,
            }
            
            logger.info(
                f"Faster-Whisper transcription 완료: "
                f"{len(result_segments)} 세그먼트, "
                f"언어: {info.language}, "
                f"길이: {info.duration:.2f}초"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription 실패: {str(e)}")
            raise
    
    @property
    def model_type(self) -> str:
        """모델 타입 반환"""
        return self._model_type.value


class STTModelFactory:
    """STT 모델 팩토리 클래스"""
    
    @staticmethod
    def create_model(
        model_type: str,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: str = "default"
    ):
        """
        STT 모델 인스턴스 생성
        
        Args:
            model_type: 모델 타입 ("whisper" 또는 "faster-whisper")
            model_size: 모델 크기
            device: 실행 디바이스
            compute_type: 연산 타입 (faster-whisper만 해당)
            
        Returns:
            모델 매니저 인스턴스
        """
        if model_type == ModelType.WHISPER.value:
            return WhisperModelManager(model_size=model_size, device=device)
        elif model_type == ModelType.FASTER_WHISPER.value:
            device = device or "cpu"
            return FasterWhisperModelManager(
                model_size=model_size,
                device=device,
                compute_type=compute_type
            )
        else:
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"'whisper' 또는 'faster-whisper'를 사용하세요."
            )


# 전역 모델 인스턴스 (싱글톤 패턴)
_global_model = None


def get_global_model():
    """전역 STT 모델 반환 (lazy loading)"""
    global _global_model
    if _global_model is None:
        # 기본 설정: Whisper medium 모델
        _global_model = STTModelFactory.create_model(
            model_type=ModelType.WHISPER.value,
            model_size="medium"
        )
        logger.info("전역 STT 모델 초기화 완료")
    return _global_model


def set_global_model(model):
    """전역 STT 모델 설정"""
    global _global_model
    _global_model = model
    logger.info(f"전역 STT 모델 변경: {model.model_type} ({model.model_size})")

