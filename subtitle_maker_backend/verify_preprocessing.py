"""
오디오 전처리 검증 도구
원본 비디오와 전처리된 오디오를 시각적/수치적으로 비교 분석
"""

import sys
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from colorama import Fore, Style, init

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if sys.platform == 'darwin' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 컬러 출력 초기화
init(autoreset=True)


def visualize_preprocessing(original_video, processed_audio):
    """
    원본 vs 전처리 비교 시각화
    
    Args:
        original_video: 원본 비디오 파일 경로
        processed_audio: 전처리된 오디오 파일 경로
    
    Returns:
        bool: 모든 검증을 통과했는지 여부
    """
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"🎵 오디오 전처리 검증 시작")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # 파일 존재 확인
    if not Path(original_video).exists():
        print(f"{Fore.RED}❌ 원본 파일을 찾을 수 없습니다: {original_video}{Style.RESET_ALL}")
        return False
    
    if not Path(processed_audio).exists():
        print(f"{Fore.RED}❌ 전처리된 파일을 찾을 수 없습니다: {processed_audio}{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.GREEN}✅ 파일 확인 완료{Style.RESET_ALL}")
    print(f"   원본: {original_video}")
    print(f"   전처리: {processed_audio}\n")
    
    # 오디오 로드
    print(f"{Fore.YELLOW}⏳ 오디오 로딩 중...{Style.RESET_ALL}")
    try:
        orig_audio, orig_sr = librosa.load(original_video, sr=None, mono=True)
        proc_audio, proc_sr = sf.read(processed_audio)
        print(f"{Fore.GREEN}✅ 오디오 로딩 완료{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}❌ 오디오 로딩 실패: {str(e)}{Style.RESET_ALL}")
        return False
    
    # 길이 맞추기 (비교를 위해)
    min_len = min(len(orig_audio), len(proc_audio))
    max_display = min(min_len, orig_sr * 30)  # 최대 30초만 표시
    
    # Figure 생성
    print(f"{Fore.YELLOW}⏳ 시각화 생성 중...{Style.RESET_ALL}")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('🎵 오디오 전처리 검증 대시보드', fontsize=18, weight='bold', y=0.995)
    
    # 1. 파형 비교
    ax1 = fig.add_subplot(gs[0, 0])
    time_orig = np.linspace(0, len(orig_audio[:max_display])/orig_sr, len(orig_audio[:max_display]))
    ax1.plot(time_orig, orig_audio[:max_display], alpha=0.7, linewidth=0.3, color='#1f77b4')
    ax1.set_title('📊 원본 오디오 파형', fontsize=12, weight='bold')
    ax1.set_xlabel('시간 (초)')
    ax1.set_ylabel('진폭')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.0, 1.0)
    
    ax2 = fig.add_subplot(gs[0, 1])
    time_proc = np.linspace(0, len(proc_audio[:max_display])/proc_sr, len(proc_audio[:max_display]))
    ax2.plot(time_proc, proc_audio[:max_display], alpha=0.7, linewidth=0.3, color='#2ca02c')
    ax2.set_title('✅ 전처리된 오디오 파형', fontsize=12, weight='bold')
    ax2.set_xlabel('시간 (초)')
    ax2.set_ylabel('진폭')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.0, 1.0)
    
    # 2. 스펙트로그램 비교
    ax3 = fig.add_subplot(gs[1, 0])
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig_audio)), ref=np.max)
    img1 = librosa.display.specshow(D_orig, sr=orig_sr, x_axis='time', y_axis='hz', ax=ax3, cmap='viridis')
    ax3.set_title('📈 원본 스펙트로그램', fontsize=12, weight='bold')
    ax3.set_ylim(0, 8000)
    plt.colorbar(img1, ax=ax3, format='%+2.0f dB')
    
    ax4 = fig.add_subplot(gs[1, 1])
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(proc_audio)), ref=np.max)
    img2 = librosa.display.specshow(D_proc, sr=proc_sr, x_axis='time', y_axis='hz', ax=ax4, cmap='viridis')
    ax4.set_title('✅ 전처리된 스펙트로그램 (노이즈 제거됨)', fontsize=12, weight='bold')
    ax4.set_ylim(0, 8000)
    plt.colorbar(img2, ax=ax4, format='%+2.0f dB')
    
    # 3. 히스토그램 (진폭 분포)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(orig_audio, bins=100, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax5.set_title('📊 원본 진폭 분포', fontsize=12, weight='bold')
    ax5.set_xlabel('진폭')
    ax5.set_ylabel('빈도')
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='중심')
    ax5.legend()
    ax5.set_xlim(-1.0, 1.0)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(proc_audio, bins=100, alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax6.set_title('✅ 전처리된 진폭 분포 (정규화됨)', fontsize=12, weight='bold')
    ax6.set_xlabel('진폭')
    ax6.set_ylabel('빈도')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='중심')
    ax6.legend()
    ax6.set_xlim(-1.0, 1.0)
    
    # 4. 통계 요약 텍스트
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    orig_rms = np.sqrt(np.mean(orig_audio**2))
    proc_rms = np.sqrt(np.mean(proc_audio**2))
    
    stats_text = f"""
    📊 수치 비교
    
    항목                    원본                    전처리                  개선
    {'─'*85}
    샘플레이트            {orig_sr:,} Hz            {proc_sr:,} Hz             {'✅ 16kHz 최적화' if proc_sr == 16000 else '❌'}
    재생 시간              {len(orig_audio)/orig_sr:.2f} 초              {len(proc_audio)/proc_sr:.2f} 초              
    RMS 레벨              {orig_rms:.4f}              {proc_rms:.4f}              {'✅ 정규화됨' if 0.05 <= proc_rms <= 0.15 else '❌'}
    최대 진폭              {np.abs(orig_audio).max():.4f}              {np.abs(proc_audio).max():.4f}              {'✅ 클리핑 없음' if np.abs(proc_audio).max() < 0.99 else '⚠️'}
    표준편차              {np.std(orig_audio):.4f}              {np.std(proc_audio):.4f}              
    """
    
    ax7.text(0.05, 0.5, stats_text, fontsize=10, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    output_file = 'preprocessing_verification.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"{Fore.GREEN}✅ 시각화 저장됨: {output_file}{Style.RESET_ALL}\n")
    
    # 그래프 표시
    plt.show()
    
    # 상세 수치 통계 출력
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"📊 상세 통계 비교")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    stats = [
        ("샘플레이트", f"{orig_sr:,} Hz", f"{proc_sr:,} Hz", proc_sr == 16000),
        ("채널", "모노", "모노", True),
        ("길이 (초)", f"{len(orig_audio)/orig_sr:.2f}", f"{len(proc_audio)/proc_sr:.2f}", True),
        ("RMS 레벨", f"{orig_rms:.4f}", f"{proc_rms:.4f}", 0.05 <= proc_rms <= 0.15),
        ("최대 진폭", f"{np.abs(orig_audio).max():.4f}", f"{np.abs(proc_audio).max():.4f}", np.abs(proc_audio).max() < 0.99),
        ("표준편차", f"{np.std(orig_audio):.4f}", f"{np.std(proc_audio):.4f}", True),
    ]
    
    print(f"{'항목':<20} {'원본':<20} {'전처리':<20} {'상태'}")
    print(f"{'-'*70}")
    for item, orig_val, proc_val, ok in stats:
        status = f"{Fore.GREEN}✅" if ok else f"{Fore.RED}❌"
        print(f"{item:<20} {orig_val:<20} {proc_val:<20} {status}{Style.RESET_ALL}")
    
    # 품질 검증
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"🔍 품질 검증")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    clipping_ratio = np.sum(np.abs(proc_audio) >= 0.99) / len(proc_audio)
    
    checks = {
        "16kHz 샘플레이트": proc_sr == 16000,
        "모노 채널": proc_audio.ndim == 1,
        "적절한 볼륨 (RMS 0.05~0.15)": 0.05 <= proc_rms <= 0.15,
        "클리핑 없음 (<1%)": clipping_ratio < 0.01,
        "무음 아님": np.abs(proc_audio).max() > 0.001,
        "파일 크기 적절": Path(processed_audio).stat().st_size > 1000,
    }
    
    all_passed = True
    for check, result in checks.items():
        status = f"{Fore.GREEN}✅ PASS" if result else f"{Fore.RED}❌ FAIL"
        print(f"{check:<40} {status}{Style.RESET_ALL}")
        if not result:
            all_passed = False
    
    # 추가 정보
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"ℹ️  추가 정보")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    file_size = Path(processed_audio).stat().st_size
    print(f"전처리 파일 크기: {file_size / (1024*1024):.2f} MB")
    print(f"클리핑 비율: {clipping_ratio * 100:.4f}%")
    print(f"다이나믹 레인지: {20*np.log10(np.abs(proc_audio).max()/(np.abs(proc_audio[proc_audio!=0]).min()+1e-10)):.1f} dB")
    
    # 최종 결과
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    if all_passed:
        print(f"{Fore.GREEN}🎉 전처리 성공! 모든 검증 통과{Style.RESET_ALL}")
        print(f"{Fore.GREEN}   전처리된 오디오가 Whisper STT에 최적화되었습니다.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  일부 검증 실패. 전처리 설정 확인이 필요합니다.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    return all_passed


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print(f"\n{Fore.YELLOW}사용법: python verify_preprocessing.py <file_id>{Style.RESET_ALL}")
        print(f"\n예시:")
        print(f"  python verify_preprocessing.py a3027232-9a43-49fe-ae72-0f9895060d70\n")
        sys.exit(1)
    
    file_id = sys.argv[1]
    
    # 파일 경로 구성
    base_dir = Path(__file__).parent / "uploads"
    
    # 원본 비디오 찾기
    original = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        candidate = base_dir / f"{file_id}{ext}"
        if candidate.exists():
            original = str(candidate)
            break
    
    if not original:
        print(f"{Fore.RED}❌ 원본 파일을 찾을 수 없습니다: {file_id}.*{Style.RESET_ALL}")
        sys.exit(1)
    
    processed = str(base_dir / f"{file_id}_processed.wav")
    
    # 검증 실행
    success = visualize_preprocessing(original, processed)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

