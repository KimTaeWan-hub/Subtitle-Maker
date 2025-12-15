"""
오디오 전처리 파이프라인 모듈
비디오 -> 오디오 추출 -> 음원 분리 -> 음성 구간 탐지
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 생성
import matplotlib.pyplot as plt

from audio_extractor import AudioProcessor
from audio_separator import AudioSeparator
from audio_detector import VoiceActivityDetector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPreprocessPipeline:
    """오디오 전처리 파이프라인 클래스"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        separator_model: str = "htdemucs",
        vad_threshold: float = 0.5,
        device: str = None
    ):
        """
        Args:
            sample_rate: 오디오 샘플 레이트 (기본값: 16000 Hz)
            separator_model: Demucs 모델 이름 (기본값: htdemucs)
            vad_threshold: VAD 음성 탐지 임계값 (0.0 ~ 1.0, 기본값: 0.5)
            device: 연산 디바이스 ('cuda', 'cpu', None=자동선택)
        """
        logger.info("AudioPreprocessPipeline 초기화 시작")
        
        # 1단계: 오디오 추출기 초기화
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        logger.info("✓ AudioProcessor 초기화 완료")
        
        # 2단계: 음원 분리기 초기화
        self.audio_separator = AudioSeparator(
            model_name=separator_model,
            device=device
        )
        logger.info("✓ AudioSeparator 초기화 완료")
        
        # 3단계: 음성 구간 탐지기 초기화
        self.voice_detector = VoiceActivityDetector(
            sample_rate=sample_rate,
            threshold=vad_threshold
        )
        logger.info("✓ VoiceActivityDetector 초기화 완료")
        
        logger.info("AudioPreprocessPipeline 초기화 완료")
    
    def visualize_preprocessing_stages(
        self,
        extracted_audio: str,
        separated_audio: str,
        final_audio: str,
        output_path: str,
        max_duration: float = 30.0
    ):
        """
        전처리 각 단계별 오디오 파형을 시각화하여 하나의 이미지로 통합
        
        Args:
            extracted_audio: 1단계 추출된 오디오 경로
            separated_audio: 2단계 분리된 오디오 경로
            final_audio: 3단계 최종 처리된 오디오 경로
            output_path: 출력 이미지 경로
            max_duration: 시각화할 최대 길이(초) - 너무 긴 파일 처리 방지
        """
        try:
            logger.info("오디오 파형 시각화 시작")
            
            # 한글 폰트 설정 (시스템 기본 폰트 사용)
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 각 단계별 오디오 로드
            audio1, sr1 = sf.read(extracted_audio)
            audio2, sr2 = sf.read(separated_audio)
            audio3, sr3 = sf.read(final_audio)
            
            # 스테레오를 모노로 변환
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=1)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=1)
            if audio3.ndim > 1:
                audio3 = np.mean(audio3, axis=1)
            
            # 최대 길이 제한 (처리 속도를 위해)
            max_samples = int(max_duration * sr1)
            if len(audio1) > max_samples:
                audio1 = audio1[:max_samples]
            if len(audio2) > max_samples:
                audio2 = audio2[:max_samples]
            if len(audio3) > max_samples:
                audio3 = audio3[:max_samples]
            
            # 시간 축 생성
            time1 = np.linspace(0, len(audio1) / sr1, len(audio1))
            time2 = np.linspace(0, len(audio2) / sr2, len(audio2))
            time3 = np.linspace(0, len(audio3) / sr3, len(audio3))
            
            # 그래프 생성 (3개의 서브플롯)
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            fig.suptitle('Audio Preprocessing Stages - Waveform', fontsize=16, fontweight='bold')
            
            # 1단계: 추출된 오디오
            axes[0].plot(time1, audio1, linewidth=0.5, color='#2E86AB')
            axes[0].set_title('Stage 1: Extracted Audio from Video', fontsize=12, pad=10)
            axes[0].set_ylabel('Amplitude', fontsize=10)
            axes[0].set_xlim([0, max(time1[-1] if len(time1) > 0 else 0, time2[-1] if len(time2) > 0 else 0, time3[-1] if len(time3) > 0 else 0)])
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim([-1, 1])
            
            # 2단계: 음원 분리된 오디오 (보컬)
            axes[1].plot(time2, audio2, linewidth=0.5, color='#A23B72')
            axes[1].set_title('Stage 2: Separated Audio (Vocals)', fontsize=12, pad=10)
            axes[1].set_ylabel('Amplitude', fontsize=10)
            axes[1].set_xlim([0, max(time1[-1] if len(time1) > 0 else 0, time2[-1] if len(time2) > 0 else 0, time3[-1] if len(time3) > 0 else 0)])
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([-1, 1])
            
            # 3단계: 최종 처리된 오디오 (음성 구간만)
            axes[2].plot(time3, audio3, linewidth=0.5, color='#F18F01')
            axes[2].set_title('Stage 3: Voice Activity Detection & Extraction', fontsize=12, pad=10)
            axes[2].set_xlabel('Time (seconds)', fontsize=10)
            axes[2].set_ylabel('Amplitude', fontsize=10)
            axes[2].set_xlim([0, max(time1[-1] if len(time1) > 0 else 0, time2[-1] if len(time2) > 0 else 0, time3[-1] if len(time3) > 0 else 0)])
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([-1, 1])
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 이미지 저장
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 파형 시각화 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"파형 시각화 실패: {str(e)}")
            return False
    
    def process(
        self,
        video_path: str,
        output_dir: str,
        extract_vocals_only: bool = True,
        detect_voice_segments: bool = True,
        keep_intermediate_files: bool = False
    ) -> Dict[str, any]:
        """
        비디오 파일의 전체 오디오 전처리 파이프라인 실행
        
        Args:
            video_path: 입력 비디오 파일 경로
            output_dir: 출력 디렉토리 경로
            extract_vocals_only: True면 보컬만 분리, False면 전체 음원 분리
            detect_voice_segments: True면 음성 구간 탐지 수행
            keep_intermediate_files: True면 중간 파일 보존, False면 최종 결과만 보존
            
        Returns:
            Dict: 처리 결과
                - extracted_audio: 추출된 오디오 경로
                - separated_audio: 분리된 오디오 경로들
                - final_audio: 최종 처리된 오디오 경로
                - voice_segments: 음성 구간 리스트 (detect_voice_segments=True인 경우)
                - statistics: 음성 통계 (detect_voice_segments=True인 경우)
        """
        try:
            # 출력 디렉토리 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            
            base_name = Path(video_path).stem
            result = {}
            
            # ===== 1단계: 오디오 추출 =====
            logger.info("\n" + "="*50)
            logger.info("1단계: 비디오에서 오디오 추출")
            logger.info("="*50)
            
            extracted_audio_path = os.path.join(output_dir, f"{base_name}_extracted.wav")
            self.audio_processor.extract_audio(video_path, extracted_audio_path)
            result['extracted_audio'] = extracted_audio_path
            
            logger.info(f"✓ 1단계 완료: {extracted_audio_path}")
            
            # ===== 2단계: 음원 분리 =====
            logger.info("\n" + "="*50)
            logger.info("2단계: 음원 분리")
            logger.info("="*50)
            
            separator_output_dir = os.path.join(output_dir, "separated")
            
            if extract_vocals_only:
                # 보컬만 분리
                separated_audio = self.audio_separator.separate_vocals(
                    extracted_audio_path,
                    separator_output_dir
                )
                # 보컬 파일을 다음 단계의 입력으로 사용
                vocals_path = separated_audio.get('vocals')
                if not vocals_path:
                    raise Exception("보컬 분리에 실패했습니다")
                stage2_output = vocals_path
            else:
                # 전체 음원 분리
                separated_audio = self.audio_separator.separate_audio(
                    extracted_audio_path,
                    separator_output_dir
                )
                # 보컬이 있으면 보컬 사용, 없으면 첫 번째 파일 사용
                stage2_output = separated_audio.get('vocals', list(separated_audio.values())[0])
            
            result['separated_audio'] = separated_audio
            logger.info(f"✓ 2단계 완료: {len(separated_audio)}개 파일 생성")
            
            # ===== 3단계: 음성 구간 탐지 =====
            if detect_voice_segments:
                logger.info("\n" + "="*50)
                logger.info("3단계: 음성 구간 탐지")
                logger.info("="*50)
                
                # 음성 구간 탐지
                voice_segments = self.voice_detector.detect_voice_segments(stage2_output)
                result['voice_segments'] = voice_segments
                
                # 통계 계산
                statistics = self.voice_detector.get_speech_statistics(
                    stage2_output,
                    voice_segments
                )
                result['statistics'] = statistics
                
                # 음성만 추출한 파일 생성
                final_audio_path = os.path.join(output_dir, f"{base_name}_final.wav")
                self.voice_detector.extract_speech_only_audio(
                    stage2_output,
                    final_audio_path,
                    voice_segments
                )
                result['final_audio'] = final_audio_path
                
                logger.info(f"✓ 3단계 완료: {len(voice_segments)}개 음성 구간 탐지")
            else:
                # 음성 구간 탐지를 하지 않는 경우, 2단계 출력을 최종 결과로 사용
                result['final_audio'] = stage2_output
                logger.info("3단계 건너뜀 (음성 구간 탐지 비활성화)")
            
            # ===== 파형 시각화 =====
            logger.info("\n파형 시각화 생성 중...")
            visualization_path = os.path.join(output_dir, f"{base_name}_waveform.png")
            self.visualize_preprocessing_stages(
                extracted_audio=extracted_audio_path,
                separated_audio=stage2_output,
                final_audio=result['final_audio'],
                output_path=visualization_path
            )
            result['visualization'] = visualization_path
            
            # ===== 중간 파일 정리 =====
            if not keep_intermediate_files:
                logger.info("\n중간 파일 정리 중...")
                
                # 추출된 원본 오디오 삭제
                if os.path.exists(extracted_audio_path):
                    os.remove(extracted_audio_path)
                    logger.info(f"삭제: {extracted_audio_path}")
                
                # 필요없는 분리된 파일 삭제
                if detect_voice_segments and extract_vocals_only:
                    # 최종 결과가 있으면 vocals 파일도 삭제
                    for stem, path in separated_audio.items():
                        if os.path.exists(path) and path != result['final_audio']:
                            os.remove(path)
                            logger.info(f"삭제: {path}")
            
            # ===== 최종 결과 출력 =====
            logger.info("\n" + "="*50)
            logger.info("전처리 파이프라인 완료")
            logger.info("="*50)
            logger.info(f"최종 오디오: {result['final_audio']}")
            
            if detect_voice_segments and 'statistics' in result:
                stats = result['statistics']
                logger.info(f"음성 구간: {stats['num_segments']}개")
                logger.info(f"전체 길이: {stats['total_duration']}초")
                logger.info(f"음성 길이: {stats['speech_duration']}초")
                logger.info(f"음성 비율: {stats['speech_ratio'] * 100:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"전처리 파이프라인 실패: {str(e)}")
            raise
    
    def process_audio_file(
        self,
        audio_path: str,
        output_dir: str,
        extract_vocals_only: bool = True,
        detect_voice_segments: bool = True,
        keep_intermediate_files: bool = False
    ) -> Dict[str, any]:
        """
        이미 추출된 오디오 파일의 전처리 파이프라인 실행 (1단계 건너뜀)
        
        Args:
            audio_path: 입력 오디오 파일 경로
            output_dir: 출력 디렉토리 경로
            extract_vocals_only: True면 보컬만 분리, False면 전체 음원 분리
            detect_voice_segments: True면 음성 구간 탐지 수행
            keep_intermediate_files: True면 중간 파일 보존, False면 최종 결과만 보존
            
        Returns:
            Dict: 처리 결과
        """
        try:
            # 출력 디렉토리 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            
            base_name = Path(audio_path).stem
            result = {}
            result['extracted_audio'] = audio_path  # 입력 파일을 extracted로 간주
            
            # ===== 2단계: 음원 분리 =====
            logger.info("\n" + "="*50)
            logger.info("2단계: 음원 분리")
            logger.info("="*50)
            
            separator_output_dir = os.path.join(output_dir, "separated")
            
            if extract_vocals_only:
                separated_audio = self.audio_separator.separate_vocals(
                    audio_path,
                    separator_output_dir
                )
                vocals_path = separated_audio.get('vocals')
                if not vocals_path:
                    raise Exception("보컬 분리에 실패했습니다")
                stage2_output = vocals_path
            else:
                separated_audio = self.audio_separator.separate_audio(
                    audio_path,
                    separator_output_dir
                )
                stage2_output = separated_audio.get('vocals', list(separated_audio.values())[0])
            
            result['separated_audio'] = separated_audio
            logger.info(f"✓ 2단계 완료: {len(separated_audio)}개 파일 생성")
            
            # ===== 3단계: 음성 구간 탐지 =====
            if detect_voice_segments:
                logger.info("\n" + "="*50)
                logger.info("3단계: 음성 구간 탐지")
                logger.info("="*50)
                
                voice_segments = self.voice_detector.detect_voice_segments(stage2_output)
                result['voice_segments'] = voice_segments
                
                statistics = self.voice_detector.get_speech_statistics(
                    stage2_output,
                    voice_segments
                )
                result['statistics'] = statistics
                
                final_audio_path = os.path.join(output_dir, f"{base_name}_final.wav")
                self.voice_detector.extract_speech_only_audio(
                    stage2_output,
                    final_audio_path,
                    voice_segments
                )
                result['final_audio'] = final_audio_path
                
                logger.info(f"✓ 3단계 완료: {len(voice_segments)}개 음성 구간 탐지")
            else:
                result['final_audio'] = stage2_output
                logger.info("3단계 건너뜀 (음성 구간 탐지 비활성화)")
            
            # ===== 파형 시각화 =====
            logger.info("\n파형 시각화 생성 중...")
            visualization_path = os.path.join(output_dir, f"{base_name}_waveform.png")
            
            # 원본 오디오는 입력 파일이므로 시각화할 수 없음, stage2_output 사용
            self.visualize_preprocessing_stages(
                extracted_audio=audio_path,
                separated_audio=stage2_output,
                final_audio=result['final_audio'],
                output_path=visualization_path
            )
            result['visualization'] = visualization_path
            
            # ===== 중간 파일 정리 =====
            if not keep_intermediate_files:
                logger.info("\n중간 파일 정리 중...")
                
                if detect_voice_segments and extract_vocals_only:
                    for stem, path in separated_audio.items():
                        if os.path.exists(path) and path != result['final_audio']:
                            os.remove(path)
                            logger.info(f"삭제: {path}")
            
            # ===== 최종 결과 출력 =====
            logger.info("\n" + "="*50)
            logger.info("전처리 파이프라인 완료")
            logger.info("="*50)
            logger.info(f"최종 오디오: {result['final_audio']}")
            
            if detect_voice_segments and 'statistics' in result:
                stats = result['statistics']
                logger.info(f"음성 구간: {stats['num_segments']}개")
                logger.info(f"전체 길이: {stats['total_duration']}초")
                logger.info(f"음성 길이: {stats['speech_duration']}초")
                logger.info(f"음성 비율: {stats['speech_ratio'] * 100:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"전처리 파이프라인 실패: {str(e)}")
            raise


# 테스트용 코드
if __name__ == "__main__":
    # 파이프라인 초기화
    pipeline = AudioPreprocessPipeline(
        sample_rate=16000,
        separator_model="htdemucs",
        vad_threshold=0.5
    )
    
    # 테스트 예제 1: 비디오 파일 처리
    test_video = "test_video.mp4"
    test_output_dir = "preprocessed_output"
    
    if os.path.exists(test_video):
        try:
            print("\n비디오 파일 전처리 시작...")
            result = pipeline.process(
                video_path=test_video,
                output_dir=test_output_dir,
                extract_vocals_only=True,
                detect_voice_segments=True,
                keep_intermediate_files=False
            )
            
            print("\n✓ 전처리 완료!")
            print(f"최종 오디오: {result['final_audio']}")
            if 'statistics' in result:
                print(f"음성 구간: {result['statistics']['num_segments']}개")
            
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
    else:
        print(f"테스트 파일이 없습니다: {test_video}")
    
    # 테스트 예제 2: 오디오 파일 처리
    test_audio = "test_audio.wav"
    
    if os.path.exists(test_audio):
        try:
            print("\n\n오디오 파일 전처리 시작...")
            result = pipeline.process_audio_file(
                audio_path=test_audio,
                output_dir=test_output_dir,
                extract_vocals_only=True,
                detect_voice_segments=True,
                keep_intermediate_files=False
            )
            
            print("\n✓ 전처리 완료!")
            print(f"최종 오디오: {result['final_audio']}")
            if 'statistics' in result:
                print(f"음성 구간: {result['statistics']['num_segments']}개")
            
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
    else:
        print(f"테스트 파일이 없습니다: {test_audio}")

