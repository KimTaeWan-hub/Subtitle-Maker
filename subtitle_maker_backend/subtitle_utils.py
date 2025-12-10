"""
자막 파일 생성 유틸리티 함수들
SRT, VTT, TXT 포맷 지원
"""


def format_time_srt(seconds: float) -> str:
    """SRT 포맷으로 시간 변환 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_time_vtt(seconds: float) -> str:
    """VTT 포맷으로 시간 변환 (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def create_srt(segments: list) -> str:
    """
    SRT 포맷 자막 파일 생성
    
    Args:
        segments: 세그먼트 리스트 [{"id": int, "start_time": float, "end_time": float, "text": str}, ...]
    
    Returns:
        SRT 포맷 문자열
    """
    srt_content = []
    for segment in segments:
        idx = segment.get("id", 0)
        start = format_time_srt(segment["start_time"])
        end = format_time_srt(segment["end_time"])
        text = segment["text"]
        
        srt_content.append(f"{idx}")
        srt_content.append(f"{start} --> {end}")
        srt_content.append(text)
        srt_content.append("")  # 빈 줄
    
    return "\n".join(srt_content)


def create_vtt(segments: list) -> str:
    """
    VTT 포맷 자막 파일 생성
    
    Args:
        segments: 세그먼트 리스트 [{"id": int, "start_time": float, "end_time": float, "text": str}, ...]
    
    Returns:
        VTT 포맷 문자열
    """
    vtt_content = ["WEBVTT", ""]
    
    for segment in segments:
        start = format_time_vtt(segment["start_time"])
        end = format_time_vtt(segment["end_time"])
        text = segment["text"]
        
        vtt_content.append(f"{start} --> {end}")
        vtt_content.append(text)
        vtt_content.append("")  # 빈 줄
    
    return "\n".join(vtt_content)


def create_txt(segments: list) -> str:
    """
    TXT 포맷 자막 파일 생성 (타임스탬프 포함)
    
    Args:
        segments: 세그먼트 리스트 [{"id": int, "start_time": float, "end_time": float, "text": str}, ...]
    
    Returns:
        TXT 포맷 문자열
    """
    txt_content = []
    
    for segment in segments:
        start = format_time_vtt(segment["start_time"])
        end = format_time_vtt(segment["end_time"])
        text = segment["text"]
        
        txt_content.append(f"[{start} --> {end}]")
        txt_content.append(text)
        txt_content.append("")  # 빈 줄
    
    return "\n".join(txt_content)

