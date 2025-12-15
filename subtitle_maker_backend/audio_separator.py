"""
오디오 음원 분리 모듈 (Demucs 사용)
"""

import os
import logging
from pathlib import Path
import torch
import demucs.api

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSeparator:
    """Demucs를 사용하여 오디오를 음원별로 분리하는 클래스"""
    
    def __init__(self, model_name: str = "htdemucs", device: str = None, segment: int = None):
        """
        Args:
            model_name: 사용할 Demucs 모델 이름 (기본값: htdemucs)
                       - htdemucs: 기본 hybrid transformer 모델
                       - mdx_extra: 고품질 MDX 모델
                       - mdx_q: 경량화된 MDX 모델
            device: 연산에 사용할 디바이스 ('cuda', 'cpu', None=자동선택)
            segment: 세그먼트 길이(초) - 메모리 부족시 줄이기
        """
        self.model_name = model_name
        
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"AudioSeparator 초기화 시작 (model: {model_name}, device: {self.device})")
        
        try:
            # Separator 초기화
            separator_kwargs = {
                "model": model_name,
                "device": self.device,
                "progress": True
            }
            
            if segment is not None:
                separator_kwargs["segment"] = segment
            
            self.separator = demucs.api.Separator(**separator_kwargs)
            
            logger.info(f"AudioSeparator 초기화 완료")
            logger.info(f"Sample rate: {self.separator.samplerate}")
            logger.info(f"Audio channels: {self.separator.audio_channels}")
            
        except Exception as e:
            logger.error(f"AudioSeparator 초기화 실패: {str(e)}")
            raise
    
    def separate_audio(self, audio_path: str, output_dir: str, stems: list = None) -> dict:
        """
        오디오 파일을 음원별로 분리
        
        Args:
            audio_path: 입력 오디오 파일 경로
            output_dir: 출력 디렉토리 경로
            stems: 분리할 음원 목록 (None이면 모두 분리)
                   예: ["vocals", "drums", "bass", "other"]
            
        Returns:
            dict: {stem_name: output_path} 형태의 딕셔너리
            
        Raises:
            Exception: 음원 분리 중 오류 발생 시
        """
        try:
            logger.info(f"음원 분리 시작: {audio_path}")
            
            # 출력 디렉토리가 없으면 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            
            # 오디오 분리 수행
            origin, separated = self.separator.separate_audio_file(audio_path)
            
            logger.info(f"분리된 음원: {list(separated.keys())}")
            
            # 결과 저장
            output_paths = {}
            base_name = Path(audio_path).stem
            
            for stem_name, source in separated.items():
                # stems 리스트가 지정되어 있고, 현재 stem이 포함되지 않으면 스킵
                if stems is not None and stem_name not in stems:
                    continue
                
                # 출력 파일 경로 생성
                output_path = os.path.join(output_dir, f"{base_name}_{stem_name}.wav")
                
                # 오디오 저장
                demucs.api.save_audio(
                    source, 
                    output_path, 
                    samplerate=self.separator.samplerate
                )
                
                output_paths[stem_name] = output_path
                logger.info(f"저장 완료: {stem_name} -> {output_path}")
            
            logger.info(f"음원 분리 완료: {len(output_paths)}개 파일 생성")
            return output_paths
            
        except Exception as e:
            logger.error(f"음원 분리 중 예외 발생: {str(e)}")
            raise
    
    def separate_vocals(self, audio_path: str, output_dir: str) -> dict:
        """
        보컬만 분리 (보컬과 반주)
        
        Args:
            audio_path: 입력 오디오 파일 경로
            output_dir: 출력 디렉토리 경로
            
        Returns:
            dict: {"vocals": vocals_path, "no_vocals": accompaniment_path}
        """
        try:
            logger.info(f"보컬 분리 시작: {audio_path}")
            
            # 출력 디렉토리가 없으면 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            
            # 오디오 분리 수행
            origin, separated = self.separator.separate_audio_file(audio_path)
            
            output_paths = {}
            base_name = Path(audio_path).stem
            
            # 보컬 저장
            if "vocals" in separated:
                vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
                demucs.api.save_audio(
                    separated["vocals"],
                    vocals_path,
                    samplerate=self.separator.samplerate
                )
                output_paths["vocals"] = vocals_path
                logger.info(f"보컬 저장 완료: {vocals_path}")
            
            # 반주 생성 (vocals를 제외한 나머지 합성)
            accompaniment = None
            for stem_name, source in separated.items():
                if stem_name != "vocals":
                    if accompaniment is None:
                        accompaniment = source
                    else:
                        accompaniment = accompaniment + source
            
            if accompaniment is not None:
                accompaniment_path = os.path.join(output_dir, f"{base_name}_no_vocals.wav")
                demucs.api.save_audio(
                    accompaniment,
                    accompaniment_path,
                    samplerate=self.separator.samplerate
                )
                output_paths["no_vocals"] = accompaniment_path
                logger.info(f"반주 저장 완료: {accompaniment_path}")
            
            logger.info(f"보컬 분리 완료")
            return output_paths
            
        except Exception as e:
            logger.error(f"보컬 분리 중 예외 발생: {str(e)}")
            raise
    
    def update_parameters(self, **kwargs):
        """
        분리 파라미터 업데이트
        
        Args:
            **kwargs: 업데이트할 파라미터들
                     예: segment=10, shifts=5
        """
        try:
            self.separator.update_parameter(**kwargs)
            logger.info(f"파라미터 업데이트 완료: {kwargs}")
        except Exception as e:
            logger.error(f"파라미터 업데이트 실패: {str(e)}")
            raise


# 테스트용 코드
if __name__ == "__main__":
    separator = AudioSeparator()
    
    # 테스트 예제
    test_audio = "test_audio.wav"
    test_output_dir = "separated_output"
    
    if os.path.exists(test_audio):
        try:
            # 전체 음원 분리
            result = separator.separate_audio(test_audio, test_output_dir)
            print(f"✓ 음원 분리 성공:")
            for stem, path in result.items():
                print(f"  - {stem}: {path}")
            
            # 보컬만 분리
            # vocals_result = separator.separate_vocals(test_audio, test_output_dir)
            # print(f"✓ 보컬 분리 성공:")
            # for stem, path in vocals_result.items():
            #     print(f"  - {stem}: {path}")
            
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
    else:
        print(f"테스트 파일이 없습니다: {test_audio}")

