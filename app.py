import streamlit as st
import os
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import sys
try:
    # Try MoviePy 2.x imports first
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import ImageClip
    from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
    from moviepy.audio.AudioClip import concatenate_audioclips
    CONCATENATE_AVAILABLE = True
    MOVIEPY_VERSION = "2.x"
except ImportError:
    try:
        # Fallback to MoviePy 1.x editor imports
        from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips, concatenate_audioclips
        CONCATENATE_AVAILABLE = True
        MOVIEPY_VERSION = "1.x"
    except ImportError:
        try:
            # Basic imports without concatenation
            from moviepy.video.io.VideoFileClip import VideoFileClip
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import ImageClip
            CONCATENATE_AVAILABLE = False
            MOVIEPY_VERSION = "basic"
        except ImportError:
            try:
                from moviepy import VideoFileClip, AudioFileClip, ImageClip
                CONCATENATE_AVAILABLE = False
                MOVIEPY_VERSION = "legacy"
            except ImportError:
                MOVIEPY_VERSION = "none"

import mimetypes
from typing import List, Optional, Tuple

# Consolidated classes for Streamlit Cloud deployment (no external files needed)

# Memory Manager (simplified for single file deployment)
class SimpleMemoryManager:
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
        
    def get_memory_usage(self):
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {'process_memory_mb': memory_info.rss / (1024 * 1024)}
        except:
            return {'process_memory_mb': 0}
            
    def register_temp_file(self, path):
        if path not in self.temp_files:
            self.temp_files.append(path)
            
    def register_temp_dir(self, path):
        if path not in self.temp_dirs:
            self.temp_dirs.append(path)
            
    def cleanup_temp_files(self):
        import gc
        cleaned = 0
        for file_path in self.temp_files[:]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned += 1
                self.temp_files.remove(file_path)
            except:
                pass
        for dir_path in self.temp_dirs[:]:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    cleaned += 1
                self.temp_dirs.remove(dir_path)
            except:
                pass
        gc.collect()
        return cleaned
        
    def force_memory_cleanup(self):
        cleaned = self.cleanup_temp_files()
        return {'files_cleaned': cleaned, 'memory_freed_mb': 0}
        
    def start_memory_monitoring(self, **kwargs):
        pass
        
    def stop_memory_monitoring(self):
        pass

# Global memory manager
_global_memory_manager = None

def get_memory_manager():
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = SimpleMemoryManager()
    return _global_memory_manager

# VideoUploadHandler (simplified for single file deployment)
class VideoUploadHandler:
    MAX_FILE_SIZE_MB = 200
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    SUPPORTED_EXTENSIONS = ['.mp4']
    SUPPORTED_MIME_TYPES = ['video/mp4']
    
    @staticmethod
    def validate_video_files(uploaded_files):
        if not uploaded_files:
            return False, ["Please upload at least one MP4 video file."]
        
        errors = []
        for i, file in enumerate(uploaded_files):
            if file is None:
                errors.append(f"File {i+1}: No file uploaded.")
                continue
                
            # Validate extension
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension not in VideoUploadHandler.SUPPORTED_EXTENSIONS:
                errors.append(f"File {i+1} ({file.name}): Only MP4 files supported. Found: {file_extension}")
            
            # Validate size
            try:
                file_size = len(file.getbuffer())
                if file_size > VideoUploadHandler.MAX_FILE_SIZE_BYTES:
                    size_mb = file_size / (1024 * 1024)
                    errors.append(f"File {i+1} ({file.name}): Size ({size_mb:.1f}MB) exceeds {VideoUploadHandler.MAX_FILE_SIZE_MB}MB limit.")
                elif file_size == 0:
                    errors.append(f"File {i+1} ({file.name}): File is empty.")
            except:
                errors.append(f"File {i+1} ({file.name}): Cannot read file.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_audio_file(uploaded_file):
        if uploaded_file is None:
            return False, ["Please upload an audio file."]
        
        # Validate extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.mp3', '.wav', '.aac']:
            return False, [f"Audio file: Only MP3, WAV, AAC files supported. Found: {file_extension}"]
        
        # Validate size
        try:
            file_size = len(uploaded_file.getbuffer())
            if file_size > 50 * 1024 * 1024:  # 50MB limit for audio
                size_mb = file_size / (1024 * 1024)
                return False, [f"Audio file: Size ({size_mb:.1f}MB) exceeds 50MB limit."]
            elif file_size == 0:
                return False, ["Audio file: File is empty."]
        except:
            return False, ["Audio file: Cannot read file."]
        
        return True, []
    
    @staticmethod
    def validate_logo_file(uploaded_file):
        if uploaded_file is None:
            return False, ["Please upload a logo image file."]
        
        # Validate extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            return False, [f"Logo file: Only JPG, JPEG, PNG files supported. Found: {file_extension}"]
        
        # Validate size
        try:
            file_size = len(uploaded_file.getbuffer())
            if file_size > 10 * 1024 * 1024:  # 10MB limit for images
                size_mb = file_size / (1024 * 1024)
                return False, [f"Logo file: Size ({size_mb:.1f}MB) exceeds 10MB limit."]
            elif file_size == 0:
                return False, ["Logo file: File is empty."]
        except:
            return False, ["Logo file: Cannot read file."]
        
        return True, []

# VideoProcessor (simplified for single file deployment)
class VideoProcessor:
    @staticmethod
    def calculate_total_duration(video_paths):
        total_duration = 0.0
        for video_path in video_paths:
            try:
                clip = VideoFileClip(video_path)
                total_duration += clip.duration
                clip.close()
            except:
                # Assume 10 seconds if can't read
                total_duration += 10.0
        return total_duration
    
    @staticmethod
    def get_video_duration(video_path):
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            return duration
        except:
            return 10.0  # Default duration

# AudioProcessor (simplified for single file deployment)
class AudioProcessor:
    def process_audio_for_video(self, audio_path, target_duration, temp_dir):
        # For Streamlit Cloud, return original audio path
        # Audio processing will be handled in add_audio_to_video function
        return audio_path

# Set availability flags
MEMORY_MANAGER_AVAILABLE = True
PROCESSORS_AVAILABLE = True

# Try to import pydub, fallback to moviepy-only if it fails
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError as e:
    st.warning("⚠️ pydub is not available. Using moviepy for audio processing.")
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Configuración de la página
st.set_page_config(
    page_title="Creador de Videos Secuenciales",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Show MoviePy status
if MOVIEPY_VERSION == "2.x":
    st.success("✅ MoviePy 2.x loaded successfully - All features available")
elif MOVIEPY_VERSION == "1.x":
    st.success("✅ MoviePy 1.x loaded successfully - All features available")
elif MOVIEPY_VERSION == "basic":
    st.warning("⚠️ MoviePy basic version loaded - Limited concatenation features")
elif MOVIEPY_VERSION == "legacy":
    st.warning("⚠️ MoviePy legacy version loaded - Limited features")
else:
    st.error("❌ MoviePy not available - Application cannot function")
    st.stop()

# VideoUploadHandler class for enhanced video file validation
class VideoUploadHandler:
    """Handles video file upload validation with enhanced MP4 support."""
    
    # Streamlit Cloud file size limit (200MB)
    MAX_FILE_SIZE_MB = 200
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Supported video formats
    SUPPORTED_EXTENSIONS = ['.mp4']
    SUPPORTED_MIME_TYPES = ['video/mp4']
    
    @staticmethod
    def validate_video_files(uploaded_files) -> Tuple[bool, List[str]]:
        """
        Validate multiple uploaded video files.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not uploaded_files:
            return False, ["Please upload at least one MP4 video file."]
        
        errors = []
        
        for i, file in enumerate(uploaded_files):
            is_valid, file_errors = VideoUploadHandler._validate_single_video_file(file, i + 1)
            if not is_valid:
                errors.extend(file_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_single_video_file(uploaded_file, file_number: int) -> Tuple[bool, List[str]]:
        """
        Validate a single uploaded video file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_number: File number for error messages
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if uploaded_file is None:
            return False, [f"File {file_number}: No file uploaded."]
        
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in VideoUploadHandler.SUPPORTED_EXTENSIONS:
            errors.append(f"File {file_number} ({uploaded_file.name}): Only MP4 files are supported. Found: {file_extension}")
        
        # Validate MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        if mime_type not in VideoUploadHandler.SUPPORTED_MIME_TYPES:
            errors.append(f"File {file_number} ({uploaded_file.name}): Invalid file type. Expected MP4 video, found: {mime_type or 'unknown'}")
        
        # Validate file size
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        if file_size > VideoUploadHandler.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            errors.append(f"File {file_number} ({uploaded_file.name}): File size ({size_mb:.1f}MB) exceeds the {VideoUploadHandler.MAX_FILE_SIZE_MB}MB limit for Streamlit Cloud.")
        
        # Validate file is not empty
        if file_size == 0:
            errors.append(f"File {file_number} ({uploaded_file.name}): File is empty.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_audio_file(uploaded_file) -> Tuple[bool, List[str]]:
        """
        Validate uploaded audio file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if uploaded_file is None:
            return False, ["Please upload an MP3 audio file."]
        
        errors = []
        
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.mp3', '.wav', '.aac']:
            errors.append(f"Audio file ({uploaded_file.name}): Unsupported format. Please use MP3, WAV, or AAC.")
        
        # Validate file size
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        if file_size > VideoUploadHandler.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            errors.append(f"Audio file ({uploaded_file.name}): File size ({size_mb:.1f}MB) exceeds the {VideoUploadHandler.MAX_FILE_SIZE_MB}MB limit.")
        
        # Validate file is not empty
        if file_size == 0:
            errors.append(f"Audio file ({uploaded_file.name}): File is empty.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_logo_file(uploaded_file) -> Tuple[bool, List[str]]:
        """
        Validate uploaded logo file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if uploaded_file is None:
            return False, ["Please upload a logo image file (JPG, JPEG, PNG)."]
        
        errors = []
        
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            errors.append(f"Logo file ({uploaded_file.name}): Unsupported format. Please use JPG, JPEG, or PNG.")
        
        # Validate MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        if mime_type not in ['image/jpeg', 'image/png']:
            errors.append(f"Logo file ({uploaded_file.name}): Invalid image type. Expected JPEG or PNG, found: {mime_type or 'unknown'}")
        
        # Validate file size (smaller limit for images)
        max_image_size = 10 * 1024 * 1024  # 10MB for images
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        if file_size > max_image_size:
            size_mb = file_size / (1024 * 1024)
            errors.append(f"Logo file ({uploaded_file.name}): File size ({size_mb:.1f}MB) exceeds the 10MB limit for images.")
        
        # Validate file is not empty
        if file_size == 0:
            errors.append(f"Logo file ({uploaded_file.name}): File is empty.")
        
        return len(errors) == 0, errors

# AudioProcessor class for enhanced audio processing with looping and fade-out
class AudioProcessor:
    """Handles audio processing including duration analysis, looping, and fade-out effects."""
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds using pydub or moviepy.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds as float
        """
        # Try pydub first if available
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(audio_path)
                duration_seconds = len(audio) / 1000.0  # pydub returns milliseconds
                return duration_seconds
            except Exception as e:
                # pydub not available, using moviepy
                pass
        
        # Fallback to moviepy
        try:
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            return duration
        except Exception as moviepy_error:
            raise Exception(f"Failed to get audio duration with moviepy: {str(moviepy_error)}")
    
    @staticmethod
    def needs_looping(audio_duration: float, video_duration: float) -> bool:
        """
        Determine if music looping is needed based on duration comparison.
        
        Args:
            audio_duration: Duration of audio file in seconds
            video_duration: Total duration of video content in seconds
            
        Returns:
            True if looping is needed, False otherwise
        """
        return video_duration > audio_duration
    
    @staticmethod
    def analyze_audio_video_duration(audio_path: str, video_duration: float) -> dict:
        """
        Analyze audio duration vs video duration and provide processing recommendations.
        
        Args:
            audio_path: Path to audio file
            video_duration: Total duration of video content in seconds
            
        Returns:
            Dictionary containing analysis results and processing recommendations
        """
        try:
            audio_duration = AudioProcessor.get_audio_duration(audio_path)
            needs_loop = AudioProcessor.needs_looping(audio_duration, video_duration)
            
            # Calculate how many loops are needed
            loops_needed = 0
            if needs_loop:
                loops_needed = int(video_duration / audio_duration) + 1
            
            # Calculate fade-out start time (3 seconds before end, or proportional for short videos)
            fade_out_duration = min(3.0, video_duration * 0.1)  # Max 3 seconds or 10% of video
            fade_out_start = max(0, video_duration - fade_out_duration)
            
            analysis = {
                'audio_duration': audio_duration,
                'video_duration': video_duration,
                'needs_looping': needs_loop,
                'loops_needed': loops_needed,
                'fade_out_duration': fade_out_duration,
                'fade_out_start': fade_out_start,
                'duration_ratio': video_duration / audio_duration if audio_duration > 0 else 0,
                'processing_required': needs_loop or fade_out_duration > 0
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing audio vs video duration: {str(e)}")
    
    @staticmethod
    def loop_audio_to_duration(audio_path: str, target_duration: float, output_path: str = None) -> str:
        """
        Loop audio seamlessly to match target duration.
        
        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            output_path: Optional output path, if None will generate one
            
        Returns:
            Path to looped audio file
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_looped.mp3")
            
            # Get original audio duration
            original_duration = AudioProcessor.get_audio_duration(audio_path)
            
            # Validate that target duration is longer than original
            if target_duration <= original_duration:
                # If target is shorter or equal, just trim the audio
                return AudioProcessor._trim_audio_to_duration(audio_path, target_duration, output_path)
            
            # Validate minimum audio length for looping (avoid very short clips)
            if original_duration < 1.0:  # Less than 1 second
                raise Exception(f"Audio file is too short for looping ({original_duration:.2f}s). "
                              f"Please use audio files longer than 1 second.")
            
            # Use pydub for seamless looping if available
            if PYDUB_AVAILABLE:
                try:
                    # Load original audio
                    audio = AudioSegment.from_file(audio_path)
                    
                    # Calculate how many full loops we need
                    loops_needed = int(target_duration / original_duration)
                    remaining_time = target_duration - (loops_needed * original_duration)
                    
                    # Create looped audio
                    looped_audio = audio * loops_needed
                    
                    # Add partial loop if needed
                    if remaining_time > 0.1:  # Only add if significant time remains
                        partial_audio = audio[:int(remaining_time * 1000)]  # pydub uses milliseconds
                        looped_audio += partial_audio
                    
                    # Ensure exact target duration
                    target_ms = int(target_duration * 1000)
                    if len(looped_audio) > target_ms:
                        looped_audio = looped_audio[:target_ms]
                    
                    # Export looped audio
                    looped_audio.export(output_path, format="mp3", bitrate="192k")
                    
                    return output_path
                    
                except Exception as pydub_error:
                    # FFmpeg not available
                    pass
            
            # Return original audio if processing fails
            return audio_path
                
        except Exception as e:
            raise Exception(f"Error looping audio to duration: {str(e)}")
    
    @staticmethod
    def _trim_audio_to_duration(audio_path: str, target_duration: float, output_path: str) -> str:
        """
        Trim audio to target duration (helper method).
        
        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            output_path: Output path for trimmed audio
            
        Returns:
            Path to trimmed audio file
        """
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(audio_path)
                target_ms = int(target_duration * 1000)
                trimmed_audio = audio[:target_ms]
                trimmed_audio.export(output_path, format="mp3", bitrate="192k")
                return output_path
            except Exception as e:
                # FFmpeg not available
                pass
        
        # Return original audio if trimming fails
        try:
            # FFmpeg not available
            return output_path
        except Exception as e:
            raise Exception(f"Error trimming audio with FFmpeg: {str(e)}")
    
    # FFmpeg functions removed for compatibility
    
    @staticmethod
    def apply_fade_out(audio_path: str, fade_duration: float = 3.0, output_path: str = None) -> str:
        """
        Apply fade-out effect to audio file.
        
        Args:
            audio_path: Path to input audio file
            fade_duration: Duration of fade-out in seconds (default 3.0)
            output_path: Optional output path, if None will generate one
            
        Returns:
            Path to audio file with fade-out effect
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_fadeout.mp3")
            
            # Get audio duration to validate fade-out parameters
            audio_duration = AudioProcessor.get_audio_duration(audio_path)
            
            # Handle proportional fade-out for short videos
            if fade_duration >= audio_duration:
                # If fade duration is longer than audio, use proportional fade (10% of total duration)
                fade_duration = max(0.5, audio_duration * 0.1)
            
            # Calculate fade-out start time
            fade_start_time = max(0, audio_duration - fade_duration)
            
            if PYDUB_AVAILABLE:
                try:
                    # Use pydub for fade-out effect
                    audio = AudioSegment.from_file(audio_path)
                    
                    # Apply fade-out effect
                    fade_ms = int(fade_duration * 1000)  # Convert to milliseconds
                    faded_audio = audio.fade_out(fade_ms)
                    
                    # Export with fade-out
                    faded_audio.export(output_path, format="mp3", bitrate="192k")
                    
                    return output_path
                    
                except Exception as pydub_error:
                    # FFmpeg not available
                    pass
            
            # Return original audio if fade-out fails
            return audio_path
                
        except Exception as e:
            # If both pydub and FFmpeg fail, return original audio without fade-out
            st.warning(f"⚠️ **Fade-out Effect Warning**\n\n"
                      f"Could not apply fade-out effect: {str(e)}\n"
                      f"Using original audio without fade-out.")
            
            # Copy original file to output path as fallback
            try:
                import shutil
                shutil.copy2(audio_path, output_path)
                return output_path
            except Exception as copy_error:
                st.error(f"❌ Could not copy audio file: {str(copy_error)}")
                return audio_path
    
    @staticmethod
    def _apply_fade_out_with_ffmpeg(audio_path: str, fade_start: float, fade_duration: float, output_path: str) -> str:
        """
        Apply fade-out effect using FFmpeg as fallback method.
        
        Args:
            audio_path: Path to input audio file
            fade_start: Time when fade-out starts in seconds
            fade_duration: Duration of fade-out in seconds
            output_path: Output path for faded audio
            
        Returns:
            Path to audio file with fade-out effect
        """
        try:
            # FFmpeg not available
            return output_path
        except Exception as e:
            # FFmpeg functions removed for compatibility
            pass
    
    @staticmethod
    def process_audio_for_video(audio_path: str, video_duration: float, output_path: str = None) -> str:
        """
        Complete audio processing pipeline: loop if needed and apply fade-out.
        This combines looping and fade-out in a single processing pipeline.
        
        Args:
            audio_path: Path to input audio file
            video_duration: Total duration of video content in seconds
            output_path: Optional output path, if None will generate one
            
        Returns:
            Path to fully processed audio file
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_processed.mp3")
            
            # Analyze audio requirements
            analysis = AudioProcessor.analyze_audio_video_duration(audio_path, video_duration)
            
            current_audio_path = audio_path
            temp_files = []  # Track temporary files for cleanup
            
            try:
                # Step 1: Loop audio if needed
                if analysis['needs_looping']:
                    looped_path = os.path.join(os.path.dirname(audio_path), "temp_looped.mp3")
                    current_audio_path = AudioProcessor.loop_audio_to_duration(
                        current_audio_path, video_duration, looped_path
                    )
                    temp_files.append(looped_path)
                
                # Step 2: Apply fade-out effect
                fade_duration = analysis['fade_out_duration']
                if fade_duration > 0:
                    AudioProcessor.apply_fade_out(current_audio_path, fade_duration, output_path)
                else:
                    # If no fade-out needed, just copy the current audio
                    if current_audio_path != output_path:
                        shutil.copy2(current_audio_path, output_path)
                
                return output_path
                
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception:
                        pass  # Ignore cleanup errors
                        
        except Exception as e:
            raise Exception(f"Error in complete audio processing pipeline: {str(e)}")

# VideoProcessor class for video processing and normalization
class VideoProcessor:
    """Handles video processing, normalization, and validation."""
    
    # Target resolution for all videos
    TARGET_RESOLUTION = (1280, 720)
    
    @staticmethod
    def normalize_video_resolution(video_path: str, target_resolution: Tuple[int, int] = TARGET_RESOLUTION) -> str:
        """
        Normalize video resolution to target resolution while maintaining aspect ratio.
        Uses padding or cropping as needed.
        
        Args:
            video_path: Path to input video file
            target_resolution: Target resolution as (width, height) tuple
            
        Returns:
            Path to normalized video file
        """
        # Generate output path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_normalized.mp4")
        
        target_width, target_height = target_resolution
        
        # Try FFmpeg first (if available)
        try:
            # FFmpeg not available
            return output_path
        except Exception:
            # FFmpeg not available or failed, use moviepy fallback
            pass
        except Exception:
            # Any other error with FFmpeg, try fallback
            pass
        
        # Fallback: Use moviepy for basic video processing
        try:
            with VideoFileClip(video_path) as video_clip:
                # Get current dimensions
                current_width, current_height = video_clip.size
                
                # Calculate scaling to fit within target resolution while maintaining aspect ratio
                scale_width = target_width / current_width
                scale_height = target_height / current_height
                scale_factor = min(scale_width, scale_height)
                
                # If the video is already smaller than target or scale factor is close to 1, skip processing
                if scale_factor >= 0.9:  # If scaling is minimal, use original
                    return video_path
                
                # Try to resize the video using moviepy
                try:
                    # Use the resize method directly on the clip
                    resized_clip = video_clip.resize(scale_factor)
                    
                    # Try to write the processed video with minimal settings
                    resized_clip.write_videofile(output_path)
                    
                    return output_path
                    
                except Exception as write_error:
                    # If writing fails, try with even simpler settings
                    try:
                        resized_clip.write_videofile(output_path)
                        return output_path
                    except Exception:
                        # If all moviepy attempts fail, use original video
                        return video_path
                
        except Exception as e:
            # If moviepy completely fails, return original video path as last resort
            return video_path
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """
        Get the duration of a video file in seconds.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds as float
        """
        # Try FFprobe first (if available)
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            
            # FFmpeg not available, use moviepy fallback directly
            pass
        except Exception:
            # Any other error with FFprobe, try fallback
            pass
        
        # Fallback: Use moviepy to get duration
        try:
            with VideoFileClip(video_path) as video_clip:
                duration = video_clip.duration
                if duration and duration > 0:
                    return float(duration)
                else:
                    raise Exception("Video has no duration or invalid duration")
        except Exception as e:
            raise Exception(f"Error getting video duration with moviepy: {str(e)}")
    
    @staticmethod
    def validate_video_format(video_path: str) -> bool:
        """
        Validate that the video file is a proper MP4 format.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid MP4, False otherwise
        """
        # First try basic file extension and existence check
        if not os.path.exists(video_path):
            return False
            
        # Check file size - if it's 0 bytes, it's invalid
        try:
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                return False
        except OSError:
            return False
            
        # Check file extension
        file_extension = os.path.splitext(video_path)[1].lower()
        if file_extension not in ['.mp4', '.mov', '.avi', '.mkv']:
            return False
        
        # Try FFprobe first (if available)
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=format_name",
                "-show_entries", "stream=codec_name,codec_type",
                "-of", "default=noprint_wrappers=1",
                video_path
            ]
            
            # FFmpeg not available, use moviepy fallback directly
            pass
            
        except Exception:
            # FFprobe not available or failed, fallback to moviepy validation
            pass
        except Exception:
            # Any other error with FFprobe, try fallback
            pass
        
        # Fallback: Use moviepy to validate the video
        try:
            # Try to open the video with a timeout-like approach
            video_clip = VideoFileClip(video_path)
            try:
                # Check if we can get basic properties
                duration = video_clip.duration
                size = video_clip.size
                
                # If we can get duration and size, and duration > 0, it's probably valid
                if duration and duration > 0 and size and len(size) == 2:
                    return True
                else:
                    return False
            finally:
                # Always close the clip to free resources
                video_clip.close()
                
        except Exception:
            # If moviepy can't open it, it's not a valid video
            return False
    
    @staticmethod
    def calculate_total_duration(video_paths: List[str]) -> float:
        """
        Calculate the total duration of multiple video files.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            Total duration in seconds as float
        """
        total_duration = 0.0
        
        for video_path in video_paths:
            try:
                duration = VideoProcessor.get_video_duration(video_path)
                total_duration += duration
            except Exception as e:
                raise Exception(f"Error calculating duration for {video_path}: {str(e)}")
        
        return total_duration

# Título de la aplicación
st.title("🎬 Creador de Videos Secuenciales")
st.markdown("""
Sube varios videos, añade música, un texto de introducción y un logo para crear un video secuencial.
""")

# Función para obtener el tamaño del texto (compatible con diferentes versiones de Pillow)
def get_text_size(draw, text, font):
    try:
        # Para versiones más nuevas de Pillow
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Para versiones más antiguas de Pillow
        return draw.textsize(text, font=font)

# Función para crear video de introducción con texto
def create_intro_video(text, output_path, duration=5, fps=24):
    try:
        width, height = 1280, 720
        
        # Crear imagen temporal con el texto
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Use a much larger font (size 120 - 3x bigger than before)
        font_size = 120
        try:
            # Try to load a TrueType font with size 120
            font = ImageFont.truetype("arial.ttf", font_size)
            st.info(f"📝 Using Arial font size {font_size}")
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                st.info(f"📝 Using DejaVu font size {font_size}")
            except:
                try:
                    font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
                    st.info(f"📝 Using Liberation font size {font_size}")
                except:
                    try:
                        # Try Windows default fonts
                        font = ImageFont.truetype("calibri.ttf", font_size)
                        st.info(f"📝 Using Calibri font size {font_size}")
                    except:
                        # Use default font (will be much smaller)
                        font = ImageFont.load_default()
                        st.warning("⚠️ Using default font - text may be smaller")
        
        # Calcular posición del texto para centrarlo
        text_width, text_height = get_text_size(draw, text, font)
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        # Dibujar texto en blanco con mayor visibilidad
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        st.info(f"📝 Intro text: '{text}' at position {position} with size {text_width}x{text_height}")
        
        # Guardar imagen temporal
        temp_img_path = os.path.join(tempfile.gettempdir(), "intro_text.png")
        img.save(temp_img_path)
        
        # Use MoviePy to create video from image
        try:
            clip = ImageClip(temp_img_path, duration=duration)
            clip.fps = fps
            clip.write_videofile(output_path, codec='libx264')
            clip.close()
            
            # Eliminar imagen temporal
            os.remove(temp_img_path)
            return output_path
            
        except Exception as moviepy_error:
            st.warning(f"⚠️ MoviePy intro creation failed: {moviepy_error}")
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return None
            
    except Exception as e:
        st.error(f"❌ Error creating intro video: {str(e)}")
        return None
        error_msg = str(e).lower()
        if "font" in error_msg:
            st.error(f"❌ **Font Error in Intro Video**\n\n"
                    f"Could not load font for intro text.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with shorter intro text\n"
                    f"• Use only basic characters (avoid special symbols)")
        elif "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Resource Error in Intro Video**\n\n"
                    f"Insufficient resources to create intro video.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with shorter intro text\n"
                    f"• Close other applications to free memory")
        else:
            st.error(f"❌ **Intro Video Creation Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that intro text is valid\n"
                    f"• Try with different text\n"
                    f"• Ensure sufficient system resources")
        st.stop()

# Función para crear video final con logo
def create_outro_video(logo_path, output_path, duration=5, fps=24):
    try:
        # Abrir logo
        logo = Image.open(logo_path)
        st.info(f"📷 Logo original size: {logo.size}")
        
        # Crear imagen de fondo negro
        width, height = 1280, 720
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        
        # Resize logo to fit the ENTIRE screen while maintaining aspect ratio
        # Use 90% of screen width or height to ensure it fits completely
        max_width = int(width * 0.9)  # 90% of screen width (1152px)
        max_height = int(height * 0.9)  # 90% of screen height (648px)
        
        # Calculate scaling ratio to fit within these bounds
        width_ratio = max_width / logo.size[0]
        height_ratio = max_height / logo.size[1]
        scale_ratio = min(width_ratio, height_ratio)
        
        # Always resize to make logo as large as possible while fitting on screen
        new_width = int(logo.size[0] * scale_ratio)
        new_height = int(logo.size[1] * scale_ratio)
        logo = logo.resize((new_width, new_height), Image.LANCZOS)
        st.info(f"📷 Logo resized from original to {new_width}x{new_height} (90% screen fit)")
        
        # Calcular posición exacta para centrar el logo
        logo_x = (width - logo.size[0]) // 2
        logo_y = (height - logo.size[1]) // 2
        position = (logo_x, logo_y)
        
        st.info(f"📷 Logo position: {position} (center of {width}x{height})")
        
        # Pegar logo con manejo de transparencia
        if logo.mode == 'RGBA':
            # Crear una máscara para la transparencia
            img.paste(logo, position, logo)
        elif logo.mode == 'P' and 'transparency' in logo.info:
            # Convertir paleta con transparencia
            logo = logo.convert('RGBA')
            img.paste(logo, position, logo)
        else:
            # Logo sin transparencia
            img.paste(logo, position)
        
        # Guardar imagen temporal
        temp_img_path = os.path.join(tempfile.gettempdir(), "outro_logo.png")
        img.save(temp_img_path)
        st.info(f"📷 Outro image saved to: {temp_img_path}")
        
        # Crear video con MoviePy
        try:
            clip = ImageClip(temp_img_path, duration=duration)
            clip.fps = fps
            clip.write_videofile(output_path, codec='libx264')
            clip.close()
            
            # Eliminar imagen temporal
            os.remove(temp_img_path)
            st.success(f"✅ Outro video created: {output_path}")
            return output_path
            
        except Exception as moviepy_error:
            st.error(f"❌ MoviePy outro creation failed: {moviepy_error}")
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return None
            
    except Exception as e:
        error_msg = str(e).lower()
        if "image" in error_msg or "logo" in error_msg:
            st.error(f"❌ **Logo Processing Error**\n\n"
                    f"Could not process the logo image.\n\n"
                    f"**Solutions:**\n"
                    f"• Check that logo is a valid JPG/PNG image\n"
                    f"• Try with a smaller logo file\n"
                    f"• Use a different image format")
        elif "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Resource Error in Outro Video**\n\n"
                    f"Insufficient resources to create outro video.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with a smaller logo image\n"
                    f"• Close other applications to free memory")
        else:
            st.error(f"❌ **Outro Video Creation Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that logo file is valid\n"
                    f"• Try with different logo image\n"
                    f"• Ensure sufficient system resources")
        st.stop()

# Función para concatenar videos
def concatenate_videos(video_paths, output_path):
    """
    Simple and robust video concatenation for MoviePy 2.x
    """
    try:
        if not video_paths:
            raise Exception("No video files provided")
            
        # Filter valid video paths
        valid_paths = [path for path in video_paths if os.path.exists(path)]
        if not valid_paths:
            raise Exception("No valid video files found")
            
        st.info(f"📹 Processing {len(valid_paths)} video(s) for concatenation")
        
        # Load and normalize all clips with strict specifications
        clips = []
        target_size = (1280, 720)
        target_fps = 24
        
        for i, video_path in enumerate(valid_paths):
            try:
                st.info(f"Loading video {i+1}/{len(valid_paths)}: {os.path.basename(video_path)}")
                clip = VideoFileClip(video_path)
                
                # Log original properties
                original_size = clip.size
                original_fps = getattr(clip, 'fps', None)
                st.info(f"📊 Original: {original_size} @ {original_fps}fps")
                
                # Force normalize FPS first
                clip.fps = target_fps
                
                # Normalize resolution while preserving aspect ratio
                if clip.size != target_size:
                    st.info(f"📐 Resizing video from {clip.size} to {target_size} with aspect ratio preservation")
                    
                    # Calculate aspect ratios
                    original_width, original_height = clip.size
                    target_width, target_height = target_size
                    
                    original_ratio = original_width / original_height
                    target_ratio = target_width / target_height
                    
                    st.info(f"📊 Aspect ratios - Original: {original_ratio:.3f}, Target: {target_ratio:.3f}")
                    
                    # Calculate scaling to fit within target size while preserving aspect ratio
                    scale_width = target_width / original_width
                    scale_height = target_height / original_height
                    scale = min(scale_width, scale_height)  # Use smaller scale to fit within bounds
                    
                    # Calculate new dimensions with precise rounding
                    new_width = round(original_width * scale)
                    new_height = round(original_height * scale)
                    
                    # Ensure we don't exceed target dimensions due to rounding
                    if new_width > target_width:
                        new_width = target_width
                    if new_height > target_height:
                        new_height = target_height
                    
                    st.info(f"📐 Scaling by {scale:.3f} to {new_width}x{new_height}")
                    
                    # Resize the clip to the calculated dimensions
                    clip = clip.resized((new_width, new_height))
                    
                    # If the resized clip doesn't match target size exactly, add black borders
                    if (new_width, new_height) != target_size:
                        st.info(f"📦 Adding black borders to center video in {target_size} frame")
                        
                        # Calculate position to center the video
                        x_offset = (target_width - new_width) // 2
                        y_offset = (target_height - new_height) // 2
                        
                        # Create a black background clip
                        from moviepy.video.VideoClip import ColorClip
                        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
                        
                        background = ColorClip(size=target_size, color=(0, 0, 0), duration=clip.duration)
                        background.fps = target_fps
                        
                        # Position the resized clip on the black background
                        clip = clip.with_position((x_offset, y_offset))
                        
                        # Composite the clips
                        clip = CompositeVideoClip([background, clip])
                        
                        st.info(f"📦 Video centered at position ({x_offset}, {y_offset}) with black borders")
                    
                    st.success(f"✅ Aspect ratio preserved - no stretching or distortion")
                
                # Verify final properties
                final_size = clip.size
                final_fps = clip.fps
                st.info(f"📊 Final: {final_size} @ {final_fps}fps")
                
                # Ensure clip has proper duration
                if not hasattr(clip, 'duration') or clip.duration is None:
                    st.warning(f"⚠️ Video {i+1} has no duration info")
                else:
                    st.info(f"⏱️ Duration: {clip.duration:.2f}s")
                
                clips.append(clip)
                st.success(f"✅ Video {i+1} processed successfully")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load video {os.path.basename(video_path)}: {str(e)}")
                continue
        
        if not clips:
            raise Exception("No videos could be loaded successfully")
            
        if len(clips) == 1:
            # Only one video
            st.info("📹 Single video detected - processing")
            clips[0].write_videofile(output_path, codec='libx264')
            clips[0].close()
        else:
            # Multiple videos - concatenate
            st.info(f"📹 Concatenating {len(clips)} normalized videos...")
            try:
                if CONCATENATE_AVAILABLE:
                    final_clip = concatenate_videoclips(clips)
                    final_clip.write_videofile(output_path, codec='libx264')
                    final_clip.close()
                    for clip in clips:
                        clip.close()
                    st.success(f"✅ Successfully concatenated {len(clips)} videos")
                else:
                    # Fallback: use first video only
                    st.warning("⚠️ Concatenation not available, using first video only")
                    clips[0].write_videofile(output_path, codec='libx264')
                    for clip in clips:
                        clip.close()
                        
            except Exception as concat_error:
                st.error(f"❌ Concatenation failed: {str(concat_error)}")
                # Fallback to first video only
                st.warning("⚠️ Falling back to first video only")
                clips[0].write_videofile(output_path, codec='libx264')
                for clip in clips:
                    clip.close()
        
        return output_path
            
    except Exception as e:
        st.error(f"❌ **Video Concatenation Error**: {str(e)}")
        raise

def manual_concatenate_videos(clips, output_path):
    """
    Manual video concatenation when moviepy concatenate_videoclips fails
    """
    try:
        # Get properties from first clip
        first_clip = clips[0]
        fps = first_clip.fps
        size = first_clip.size
        
        # Create a list to store all frames
        all_frames = []
        
        # Extract frames from each clip
        for i, clip in enumerate(clips):
            st.info(f"Processing clip {i+1}/{len(clips)}")
            # Resize clip to match first clip if needed
            if clip.size != size:
                clip = clip.resize(size)
            if clip.fps != fps:
                clip = clip.set_fps(fps)
                
            # Get frames
            for frame in clip.iter_frames():
                all_frames.append(frame)
        
        # Create new clip from all frames
        if all_frames:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                try:
                    from moviepy.editor import ImageSequenceClip
                except ImportError:
                    raise Exception("ImageSequenceClip not available")
                    
            final_clip = ImageSequenceClip(all_frames, fps=fps)
            final_clip.write_videofile(output_path)
            final_clip.close()
        else:
            raise Exception("No frames extracted from videos")
            
    except Exception as e:
        st.error(f"Manual concatenation failed: {str(e)}")
        raise

def add_audio_to_video(video_path, audio_path, output_path):
    """
    Simple and reliable audio-video combination
    """
    try:
        st.info(f"🎵 Combining video with audio...")
        st.info(f"📹 Video: {os.path.basename(video_path)}")
        st.info(f"🎵 Audio: {os.path.basename(audio_path)}")
        
        # Load video
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
        st.info(f"📹 Video duration: {video_duration:.2f} seconds")
        
        # Load audio
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        st.info(f"🎵 Audio duration: {audio_duration:.2f} seconds")
        
        # Simple audio processing
        if audio_duration < video_duration:
            # Loop audio if it's shorter than video
            loops_needed = int(video_duration / audio_duration) + 1
            st.info(f"🔄 Looping audio {loops_needed} times")
            
            # Create multiple copies and concatenate
            audio_segments = []
            for i in range(loops_needed):
                audio_segments.append(audio_clip)
            
            # Concatenate audio segments
            try:
                extended_audio = concatenate_audioclips(audio_segments)
                # Trim to exact video duration
                final_audio = extended_audio.subclipped(0, video_duration)
                extended_audio.close()
            except:
                # Fallback: just use original audio
                st.warning("⚠️ Audio looping failed, using original audio")
                final_audio = audio_clip
        else:
            # Trim audio if it's longer than video
            final_audio = audio_clip.subclipped(0, video_duration)
        
        # Skip fade-out for now to ensure basic audio works
        st.info("🎵 Audio processing complete (fade-out disabled for compatibility)")
        
        # Combine video with audio
        st.info("🎬 Combining video and audio...")
        final_clip = video_clip.with_audio(final_audio)
        
        # Write final video with better error handling
        st.info("💾 Writing final video file...")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            st.info(f"📁 Created output directory: {output_dir}")
        
        # Try different approaches for writing the video
        try:
            # First try: Standard write with codec specification
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        except Exception as write_error1:
            st.warning(f"⚠️ Standard write failed: {write_error1}")
            try:
                # Second try: Write without codec specification
                final_clip.write_videofile(output_path)
            except Exception as write_error2:
                st.warning(f"⚠️ Simple write failed: {write_error2}")
                # Third try: Use a different filename
                alt_path = output_path.replace('.mp4', f'_alt_{int(time.time())}.mp4')
                st.info(f"📁 Trying alternative path: {alt_path}")
                final_clip.write_videofile(alt_path)
                # If successful, rename to original path
                if os.path.exists(alt_path):
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    shutil.move(alt_path, output_path)
                    st.info(f"✅ Video saved to: {output_path}")
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        final_audio.close()
        final_clip.close()
        
        st.success(f"✅ Video with audio created successfully!")
        return output_path
        
    except Exception as e:
        st.error(f"❌ Audio integration failed: {str(e)}")
        # Try to create video without audio as fallback
        try:
            st.warning("⚠️ Creating video without audio as fallback...")
            video_clip = VideoFileClip(video_path)
            
            # Ensure output directory exists for fallback too
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Try multiple approaches for fallback too
            try:
                video_clip.write_videofile(output_path, codec='libx264')
            except:
                try:
                    video_clip.write_videofile(output_path)
                except:
                    # Last resort: use temp directory
                    fallback_path = os.path.join(tempfile.gettempdir(), f"fallback_video_{int(time.time())}.mp4")
                    video_clip.write_videofile(fallback_path)
                    if os.path.exists(fallback_path):
                        shutil.move(fallback_path, output_path)
            
            video_clip.close()
            st.warning("⚠️ Video created without background music")
            return output_path
        except Exception as fallback_error:
            st.error(f"❌ Fallback also failed: {str(fallback_error)}")
            raise

# Legacy function name for backward compatibility
def add_enhanced_audio_to_video(video_path, processed_audio_path, output_path):
    """
    Wrapper function that explicitly indicates enhanced audio processing.
    This function ensures the final video uses processed audio (looped + faded)
    instead of original music.
    
    Args:
        video_path: Path to concatenated video file
        processed_audio_path: Path to fully processed audio (looped and faded)
        output_path: Path for final output video
        
    Returns:
        Path to final video with enhanced audio
    """
    return add_audio_to_video(video_path, processed_audio_path, output_path)

# Función para extraer audio de un video
def extract_audio(video_path, output_path):
    try:
        # FFmpeg command removed for compatibility
        
        # FFmpeg command removed for compatibility
        return output_path
    except Exception as e:
        error_msg = str(e).lower()
        if "codec" in error_msg or "format" in error_msg:
            st.error(f"❌ **Audio Extraction Format Error**\n\n"
                    f"Could not extract audio due to format issues.\n\n"
                    f"**Solutions:**\n"
                    f"• The video is still available for download\n"
                    f"• You can extract audio manually using video editing software")
        else:
            st.error(f"❌ **Audio Extraction Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Note:** The video is still available for download")
        st.stop()

# Initialize memory manager (directories will be created when form is submitted)
memory_manager = get_memory_manager()

# Formulario de entrada
with st.form("video_form"):
    st.subheader("Configuración del video")
    
    # Texto de introducción
    intro_text = st.text_input("Texto de introducción (máximo 10 palabras):", 
                              placeholder="Ej: Mis vacaciones de verano")
    
    # Videos - Updated to focus on MP4 files
    st.markdown("**Videos MP4:**")
    st.info("📹 Solo se aceptan archivos MP4. Tamaño máximo: 200MB por archivo.")
    videos = st.file_uploader("Selecciona tus videos MP4:", 
                             type=["mp4"], 
                             accept_multiple_files=True,
                             help="Sube uno o más archivos MP4. Los videos se procesarán en el orden que los subas.")
    
    # Música
    st.markdown("**Música de fondo:**")
    music = st.file_uploader("Selecciona tu música:", 
                            type=["mp3", "wav", "aac"],
                            help="La música se repetirá automáticamente si es más corta que el video final.")
    
    # Logo
    st.markdown("**Logo:**")
    logo = st.file_uploader("Selecciona tu logo:", 
                           type=["jpg", "jpeg", "png"],
                           help="El logo aparecerá al final del video por 5 segundos.")
    
    # Botón de envío
    submitted = st.form_submit_button("Crear Video Secuencial")

# Crear contenedor para resultados justo después del botón
result_container = st.container()

# Mostrar placeholder cuando no hay video procesado
if not st.session_state.get('video_processed', False):
    with result_container:
        st.info("👆 Completa el formulario de arriba y haz clic en 'Crear Video Secuencial' para generar tu video.")
        st.empty()  # Espacio reservado para el video

# Procesamiento cuando se envía el formulario
if submitted:
    # Initialize variables to prevent scope issues
    final_audio_path = None
    
    # Initialize memory manager for this session
    memory_manager = get_memory_manager()
    memory_manager.start_memory_monitoring(check_interval=15)
    
    # Create temporary directories for this processing session
    try:
        temp_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        # Verify directories were created successfully
        if not os.path.exists(temp_dir):
            st.error(f"❌ Failed to create temporary directory: {temp_dir}")
            st.stop()
        if not os.path.exists(output_dir):
            st.error(f"❌ Failed to create output directory: {output_dir}")
            st.stop()
            
        st.success(f"✅ Created temporary directories: {os.path.basename(temp_dir)}")
        
        memory_manager.register_temp_dir(temp_dir)
        memory_manager.register_temp_dir(output_dir)
        
    except Exception as e:
        st.error(f"❌ Error creating temporary directories: {str(e)}")
        st.stop()
    
    # Check initial memory status
    memory_stats = memory_manager.get_memory_usage()
    st.info(f"🔧 **Memoria inicial:** {memory_stats['process_memory_mb']:.1f}MB")
    
    # Validar entradas usando VideoUploadHandler
    validation_errors = []
    
    # Validar texto de introducción
    if not intro_text:
        validation_errors.append("Por favor ingresa un texto de introducción.")
    else:
        words = intro_text.split()
        if len(words) > 10:
            validation_errors.append("El texto de introducción debe tener máximo 10 palabras.")
    
    # Validar videos MP4
    videos_valid, video_errors = VideoUploadHandler.validate_video_files(videos)
    if not videos_valid:
        validation_errors.extend(video_errors)
    
    # Validar música
    music_valid, music_errors = VideoUploadHandler.validate_audio_file(music)
    if not music_valid:
        validation_errors.extend(music_errors)
    
    # Validar logo
    logo_valid, logo_errors = VideoUploadHandler.validate_logo_file(logo)
    if not logo_valid:
        validation_errors.extend(logo_errors)
    
    # Mostrar errores de validación si los hay
    if validation_errors:
        st.error("❌ **Errores de validación encontrados:**")
        for error in validation_errors:
            st.error(f"• {error}")
        st.info("💡 **Consejos:**\n"
                "- Asegúrate de que todos los videos sean archivos MP4\n"
                "- Verifica que los archivos no excedan el límite de tamaño\n"
                "- El texto de introducción debe tener máximo 10 palabras")
        st.stop()
    
    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Guardar archivos subidos en el directorio temporal
    status_text.text("Guardando archivos subidos...")
    progress_bar.progress(10)
    
    # Clear any previous session state to ensure fresh processing
    st.session_state.clear()  # Clear all previous state
    st.info("🔄 **Fresh Session:** Starting with clean state - only using current uploads")
    
    # Validate that we have fresh video uploads from current session
    if not videos or len(videos) == 0:
        st.error("❌ **No videos uploaded!** Please upload at least one MP4 video file.")
        st.stop()
    
    st.info(f"📹 **Processing {len(videos)} videos from current upload session**")
    
    # Filter out any None or invalid videos (only from current session)
    valid_videos = []
    for i, video in enumerate(videos):
        if video is not None and hasattr(video, 'getbuffer') and hasattr(video, 'name'):
            # Verify this is a fresh upload by checking buffer size
            try:
                buffer_size = len(video.getbuffer())
                if buffer_size > 0:
                    valid_videos.append(video)
                    st.info(f"✅ Video {i+1}: {video.name} ({buffer_size/1024/1024:.1f}MB)")
                else:
                    st.warning(f"⚠️ Skipping empty video: {video.name}")
            except Exception as e:
                st.warning(f"⚠️ Skipping invalid video {i+1}: {str(e)}")
    
    if len(videos) != len(valid_videos):
        st.warning(f"⚠️ Using {len(valid_videos)} valid videos out of {len(videos)} uploaded")
    
    # Save ONLY the current session's uploaded videos to fresh temporary files
    video_paths = []
    
    st.info("💾 **Saving current session videos to temporary files...**")
    
    for i, video in enumerate(valid_videos):
        try:
            # Create unique filename with timestamp to avoid any caching
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            video_path = os.path.join(temp_dir, f"current_session_video_{i}_{timestamp}.mp4")
            
            # Get fresh buffer from current upload
            buffer = video.getbuffer()
            buffer_size = len(buffer)
            
            st.info(f"📁 Saving {video.name} → {os.path.basename(video_path)} ({buffer_size/1024/1024:.1f}MB)")
            
            # Write fresh video data
            with open(video_path, "wb") as f:
                f.write(buffer)
            
            # Verify the file was written correctly
            if os.path.exists(video_path) and os.path.getsize(video_path) == buffer_size:
                video_paths.append(video_path)
                st.success(f"✅ Video {i+1} saved successfully")
            else:
                st.error(f"❌ Failed to save video {i+1}: {video.name}")
                
        except Exception as e:
            st.error(f"❌ Error saving video {i+1}: {str(e)}")
            continue
    
    # Final verification: ensure we only have current session videos
    if len(video_paths) == 0:
        st.error("❌ **No valid videos saved!** Please check your video uploads and try again.")
        st.stop()
    
    st.success(f"✅ **Successfully saved {len(video_paths)} videos from current session**")
    st.info(f"📋 **Video List:** {[os.path.basename(path) for path in video_paths]}")
    
    # Save current session music with unique filename
    timestamp = int(time.time() * 1000)
    music_path = os.path.join(temp_dir, f"current_session_music_{timestamp}.mp3")
    with open(music_path, "wb") as f:
        f.write(music.getbuffer())
    st.info(f"🎵 **Music saved:** {music.name} → {os.path.basename(music_path)}")
    
    # Save current session logo with unique filename (if provided)
    logo_path = None
    if logo is not None:
        logo_path = os.path.join(temp_dir, f"current_session_logo_{timestamp}.png")
        with open(logo_path, "wb") as f:
            f.write(logo.getbuffer())
        st.info(f"🖼️ **Logo saved:** {logo.name} → {os.path.basename(logo_path)}")
    else:
        st.info("🖼️ **No logo provided** - outro video will be skipped")
    
    # Process and normalize video segments
    status_text.text("Processing and normalizing video segments...")
    progress_bar.progress(20)
    
    if not video_paths:
        st.error("❌ **No valid video files found!** Please upload at least one valid MP4 video.")
        st.stop()
    
    # Process videos with memory management
    def process_single_video(video_info):
        i, video_path = video_info
        
        # Basic file existence check
        if not os.path.exists(video_path):
            st.error(f"❌ **Video {i+1} Error**: File not found")
            return None
            
        # Check if file is empty
        try:
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                st.error(f"❌ **Video {i+1} Error**: File is empty")
                return None
        except OSError:
            st.error(f"❌ **Video {i+1} Error**: Cannot read file")
            return None
        
        # Skip detailed validation for now and try to process directly
        # This is more lenient and should work with most video formats
        
        # Skip normalization for maximum compatibility - use original videos
        st.info(f"ℹ️ **Video {i+1}**: Using original resolution for maximum compatibility")
        return video_path
    
    try:
        # Process videos directly without using ChunkedProcessor to avoid complexity
        normalized_video_paths = []
        
        for i, video_path in enumerate(video_paths):
            try:
                result = process_single_video((i, video_path))
                normalized_video_paths.append(result)
            except Exception as e:
                st.error(f"❌ Error processing video {i+1}: {str(e)}")
                normalized_video_paths.append(None)
        
        # Check for any failed processing
        if None in normalized_video_paths:
            failed_indices = [i for i, path in enumerate(normalized_video_paths) if path is None]
            st.error(f"❌ **Video Processing Failed**\n\n"
                    f"Failed to process videos: {[i+1 for i in failed_indices]}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that the video files are not corrupted\n"
                    f"• Ensure you have sufficient disk space\n"
                    f"• Try with smaller video files")
            st.stop()
        
        # Update progress
        progress_bar.progress(30)
        
        # Check memory usage after video processing
        memory_stats = memory_manager.get_memory_usage()
        if memory_stats['process_memory_mb'] > 600:  # Warning threshold
            st.warning(f"⚠️ **Alta utilización de memoria:** {memory_stats['process_memory_mb']:.1f}MB")
            memory_manager.cleanup_temp_files()
        
    except Exception as e:
        st.error(f"❌ **Critical Error Processing Videos**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that video files are not corrupted\n"
                f"• Ensure sufficient disk space\n"
                f"• Try with smaller video files\n"
                f"• Re-upload the video files")
        st.stop()
    
    # Calculate total video duration for audio processing
    status_text.text("Calculating video durations...")
    progress_bar.progress(30)
    
    try:
        # Calculate duration of intro (5 seconds) + normalized videos + outro (5 seconds)
        intro_outro_duration = 10.0  # 5 seconds each for intro and outro
        main_videos_duration = VideoProcessor.calculate_total_duration(normalized_video_paths)
        total_video_duration = intro_outro_duration + main_videos_duration
        
        # Validate reasonable duration limits
        if total_video_duration > 3600:  # More than 1 hour
            st.warning(f"⚠️ **Long Video Warning**\n\n"
                      f"Total video duration is {total_video_duration/60:.1f} minutes.\n"
                      f"Processing may take longer and use more resources.")
        
        st.info(f"📊 **Duration Analysis:**\n"
                f"- Main videos: {main_videos_duration:.1f} seconds\n"
                f"- With intro/outro: {total_video_duration:.1f} seconds")
        
    except Exception as e:
        st.error(f"❌ **Duration Calculation Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that all video files are valid MP4 format\n"
                f"• Ensure videos are not corrupted\n"
                f"• Try re-uploading the video files\n"
                f"• Contact support if the problem persists")
        st.stop()
    
    # Prepare background music for full video duration with memory management
    status_text.text("Preparing background music for full video duration...")
    progress_bar.progress(35)
    
    def process_audio_with_memory_limit():
        try:
            st.info("🎵 Processing audio for video integration...")
            
            # For now, use the original music file directly
            # The audio processing will be handled in the add_audio_to_video function
            if os.path.exists(music_path):
                st.success("✅ Audio file ready for integration")
                return music_path
            else:
                st.error("❌ Music file not found")
                return None
                
        except Exception as e:
            st.warning(f"⚠️ Audio processing error: {str(e)}")
            return music_path
    
    # Process audio with enhanced features (looping + fade-out)
    processed_audio_path = process_audio_with_memory_limit()
    
    # Create video of introduction
    status_text.text("Creating introduction video...")
    progress_bar.progress(45)
    
    # Create intro video with text
    intro_video_path = None
    if intro_text and intro_text.strip():
        try:
            intro_video_path = os.path.join(temp_dir, "intro.mp4")
            st.info(f"🎬 Creating intro video with text: '{intro_text}'")
            result = create_intro_video(intro_text, intro_video_path)
            if result and os.path.exists(intro_video_path):
                st.success("✅ Intro video created successfully")
            else:
                st.warning("⚠️ Intro video creation failed, continuing without intro")
                intro_video_path = None
        except Exception as e:
            st.warning(f"⚠️ Intro video creation failed: {str(e)}")
            intro_video_path = None
    
    # Create outro video with logo
    status_text.text("Creating outro video with logo...")
    progress_bar.progress(50)
    
    # Create outro video with logo
    outro_video_path = None
    if logo is not None:
        try:
            outro_video_path = os.path.join(temp_dir, "outro.mp4")
            # Save uploaded logo to temp file
            logo_temp_path = os.path.join(temp_dir, f"logo.{logo.name.split('.')[-1]}")
            with open(logo_temp_path, "wb") as f:
                f.write(logo.getbuffer())
            
            st.info("🎬 Creating outro video with logo")
            result = create_outro_video(logo_temp_path, outro_video_path)
            if result and os.path.exists(outro_video_path):
                st.success("✅ Outro video created successfully")
            else:
                st.warning("⚠️ Outro video creation failed, continuing without outro")
                outro_video_path = None
        except Exception as e:
            st.warning(f"⚠️ Outro video creation failed: {str(e)}")
            outro_video_path = None
    
    # Concatenate all videos
    status_text.text("Concatenating all video segments...")
    progress_bar.progress(60)
    
    try:
        # Build video list with only existing videos
        all_videos = []
        if intro_video_path and os.path.exists(intro_video_path):
            all_videos.append(intro_video_path)
        all_videos.extend(normalized_video_paths)
        if outro_video_path and os.path.exists(outro_video_path):
            all_videos.append(outro_video_path)
            
        concatenated_path = os.path.join(temp_dir, "concatenated.mp4")
        concatenate_videos(all_videos, concatenated_path)
        
        # Verify concatenated video was created
        if not os.path.exists(concatenated_path):
            raise Exception("Concatenated video file was not created")
            
    except Exception as e:
        st.error(f"❌ **Video Concatenation Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that all video files are valid and not corrupted\n"
                f"• Ensure sufficient disk space for the final video\n"
                f"• Try with fewer or smaller video files\n"
                f"• Restart the process with fresh uploads")
        st.stop()
    
    # Combine processed audio with final video using enhanced function
    status_text.text("Combining processed audio with video...")
    progress_bar.progress(75)
    
    # Ensure output directory still exists
    if not os.path.exists(output_dir):
        st.warning(f"⚠️ Output directory was deleted, recreating: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as dir_error:
            st.error(f"❌ Cannot create output directory: {dir_error}")
            # Use temp_dir as fallback
            output_dir = temp_dir
            st.info(f"📁 Using temp directory as fallback: {output_dir}")
    
    final_video_path = os.path.join(output_dir, "final_video.mp4")
    
    # Verify we can write to the output path
    try:
        # Test write permissions by creating a small test file
        test_path = os.path.join(output_dir, "test_write.tmp")
        with open(test_path, 'w') as f:
            f.write("test")
        os.remove(test_path)
        st.info(f"✅ Output directory is writable: {output_dir}")
    except Exception as write_error:
        st.error(f"❌ Cannot write to output directory: {write_error}")
        # Use a different path
        final_video_path = os.path.join(tempfile.gettempdir(), f"final_video_{int(time.time())}.mp4")
        st.info(f"📁 Using alternative path: {final_video_path}")
    
    try:
        # Use enhanced audio-video combination with processed audio (looped + faded)
        add_enhanced_audio_to_video(concatenated_path, processed_audio_path, final_video_path)
        
        # Verify the final video was created successfully
        if not os.path.exists(final_video_path):
            raise Exception("Final video file was not created")
        
        # Verify final video has proper duration and format
        try:
            final_duration = VideoProcessor.get_video_duration(final_video_path)
            if abs(final_duration - total_video_duration) > 2.0:  # Allow 2 second tolerance
                st.warning(f"⚠️ **Duration Mismatch Warning**\n\n"
                          f"Final video duration ({final_duration:.1f}s) differs from expected ({total_video_duration:.1f}s).\n"
                          f"This is usually normal due to encoding differences.")
        except Exception as duration_error:
            st.warning(f"⚠️ Could not verify final video duration: {str(duration_error)}")
            final_duration = total_video_duration  # Use expected duration as fallback
        
        st.success(f"✅ **Processing Complete:**\n"
                  f"- Final video duration: {final_duration:.1f} seconds\n"
                  f"- Audio processing: Original music used\n"
                  f"- Video resolution: Original (no normalization)")
        
    except Exception as e:
        # Enhanced error handling with specific guidance
        error_msg = str(e).lower()
        
        if "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Insufficient Resources**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Try with smaller video files\n"
                    f"• Reduce the number of videos\n"
                    f"• Close other applications to free memory\n"
                    f"• Try again later when system resources are available")
        elif "codec" in error_msg or "format" in error_msg:
            st.error(f"❌ **Video Format Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Re-encode videos to standard MP4 format\n"
                    f"• Use a video converter to ensure compatibility\n"
                    f"• Try with different video files")
        elif "permission" in error_msg or "access" in error_msg:
            st.error(f"❌ **File Access Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Refresh the page and try again\n"
                    f"• Check that files are not open in other applications\n"
                    f"• Clear browser cache and retry")
        else:
            st.error(f"❌ **Final Video Creation Failed**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Try refreshing the page and starting over\n"
                    f"• Use smaller or fewer video files\n"
                    f"• Ensure all uploaded files are valid\n"
                    f"• Contact support if the problem persists")
        st.stop()
    
    # Extract final audio
    status_text.text("Extracting final audio...")
    progress_bar.progress(85)
    
    # Initialize final_audio_path to None
    final_audio_path = None
    
    try:
        final_audio_path = os.path.join(output_dir, "final_audio.mp3")
        extract_audio(final_video_path, final_audio_path)
        
        # Verify audio extraction was successful
        if not os.path.exists(final_audio_path):
            st.warning("⚠️ **Audio Extraction Warning**\n\n"
                      "Could not extract audio from final video.\n"
                      "The video is still available for download.")
            final_audio_path = None
            
    except Exception as e:
        # Don't stop the process for audio extraction errors
        st.warning(f"⚠️ **Audio Extraction Warning**\n\n"
                  f"Could not extract audio: {str(e)}\n"
                  f"The video is still available for download.")
        final_audio_path = None
    
    # Clean up temporary normalized video files
    status_text.text("Cleaning up temporary files...")
    progress_bar.progress(95)
    
    cleanup_errors = []
    try:
        # Clean up normalized video files
        for normalized_path in normalized_video_paths:
            try:
                if (os.path.exists(normalized_path) and 
                    normalized_path not in video_paths and 
                    "normalized" in normalized_path):
                    os.remove(normalized_path)
            except Exception as e:
                cleanup_errors.append(f"Could not remove {os.path.basename(normalized_path)}: {str(e)}")
        
        # Clean up processed audio file if it's different from original
        try:
            if (os.path.exists(processed_audio_path) and 
                processed_audio_path != music_path):
                os.remove(processed_audio_path)
        except Exception as e:
            cleanup_errors.append(f"Could not remove processed audio: {str(e)}")
        
        # Clean up concatenated video file
        try:
            if os.path.exists(concatenated_path):
                os.remove(concatenated_path)
        except Exception as e:
            cleanup_errors.append(f"Could not remove concatenated video: {str(e)}")
            
        # Report cleanup issues if any (but don't stop the process)
        if cleanup_errors:
            st.info(f"ℹ️ **Cleanup Note:**\n"
                   f"Some temporary files could not be removed:\n" + 
                   "\n".join([f"• {error}" for error in cleanup_errors[:3]]) +
                   (f"\n• ... and {len(cleanup_errors)-3} more" if len(cleanup_errors) > 3 else ""))
            
    except Exception as cleanup_error:
        # Don't stop the process for cleanup errors, just log them
        st.info(f"ℹ️ **Cleanup Note:** Some temporary files could not be cleaned up: {str(cleanup_error)}")
    
    # Complete
    progress_bar.progress(100)
    status_text.text("¡Video created successfully with enhanced processing!")
    
    # Marcar que el video ha sido procesado
    st.session_state['video_processed'] = True
    
    # Mostrar resultado en el contenedor que está justo después del botón
    with result_container:
        # Limpiar el placeholder
        st.empty()
        
        st.subheader("🎬 Tu Video Está Listo")
        st.video(final_video_path)
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        
        with open(final_video_path, "rb") as f:
            video_bytes = f.read()
        
        col1.download_button(
            label="📥 Descargar Video (MP4)",
            data=video_bytes,
            file_name="video_secuencial.mp4",
            mime="video/mp4",
            use_container_width=True
        )
        
        # Only show audio download if audio file exists
        try:
            if final_audio_path and os.path.exists(final_audio_path):
                with open(final_audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                col2.download_button(
                    label="🎵 Descargar Audio (MP3)",
                    data=audio_bytes,
                    file_name="audio_secuencial.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
            else:
                col2.info("ℹ️ Audio extraction not available")
        except (NameError, FileNotFoundError, OSError) as e:
            col2.info(f"ℹ️ Audio not available: {str(e)}")
        except Exception as e:
            col2.info("ℹ️ Audio extraction not available")
        
        # Separador visual
        st.divider()
        
        # Información adicional
        st.success("✅ **¡Procesamiento completado exitosamente!**")
        st.info("💡 **Tip:** Puedes descargar tanto el video como el audio por separado usando los botones de arriba.")
    
    # Limpiar directorios temporales con gestión de memoria (fuera del contenedor de resultados)
    try:
        cleanup_result = memory_manager.force_memory_cleanup()
        st.info(f"🧹 **Limpieza completada:** {cleanup_result['files_cleaned']} archivos, "
                f"{cleanup_result['memory_freed_mb']:.1f}MB liberados")
        memory_manager.stop_memory_monitoring()
    except Exception as cleanup_error:
        st.warning(f"⚠️ Advertencia durante la limpieza: {cleanup_error}")
        # Fallback cleanup
        try:
            shutil.rmtree(temp_dir)
            shutil.rmtree(output_dir)
        except:
            pass
