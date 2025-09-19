import streamlit as st
import os
import subprocess
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Creador de Videos Secuenciales",
    page_icon="🎬",
    layout="centered"
)

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
def create_intro_video(text, output_path, duration=3, fps=24):
    width, height = 1280, 720
    
    # Crear imagen temporal con el texto
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Usar fuente predeterminada
    font = ImageFont.load_default()
    
    # Calcular posición del texto para centrarlo
    text_width, text_height = get_text_size(draw, text, font)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Dibujar texto
    draw.text(position, text, fill=(255, 255, 255), font=font)
    
    # Guardar imagen temporal
    temp_img_path = os.path.join(tempfile.gettempdir(), "intro_text.png")
    img.save(temp_img_path)
    
    # Crear video con FFmpeg
    cmd = [
        "ffmpeg",
        "-loop", "1",
        "-i", temp_img_path,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # Eliminar imagen temporal
    os.remove(temp_img_path)
    
    return output_path

# Función para crear video final con logo
def create_outro_video(logo_path, output_path, duration=3, fps=24):
    # Abrir logo
    logo = Image.open(logo_path)
    
    # Redimensionar logo si es necesario
    max_size = 400
    if max(logo.size) > max_size:
        ratio = max_size / max(logo.size)
        logo = logo.resize((int(logo.size[0] * ratio), int(logo.size[1] * ratio)), Image.LANCZOS)
    
    # Crear imagen de fondo
    width, height = 1280, 720
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    
    # Calcular posición del logo para centrarlo
    position = ((width - logo.size[0]) // 2, (height - logo.size[1]) // 2)
    
    # Pegar logo
    img.paste(logo, position)
    
    # Guardar imagen temporal
    temp_img_path = os.path.join(tempfile.gettempdir(), "outro_logo.png")
    img.save(temp_img_path)
    
    # Crear video con FFmpeg
    cmd = [
        "ffmpeg",
        "-loop", "1",
        "-i", temp_img_path,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # Eliminar imagen temporal
    os.remove(temp_img_path)
    
    return output_path

# Función para concatenar videos
def concatenate_videos(video_paths, output_path):
    # Crear archivo de lista para ffmpeg
    list_file = os.path.join(tempfile.gettempdir(), "file_list.txt")
    with open(list_file, "w") as f:
        for video_path in video_paths:
            f.write(f"file '{video_path}'\n")
    
    # Usar ffmpeg para concatenar
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    os.remove(list_file)
    return output_path

# Función para añadir audio a un video
def add_audio_to_video(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

# Función para extraer audio de un video
def extract_audio(video_path, output_path):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

# Crear directorios temporales
temp_dir = tempfile.mkdtemp()
output_dir = tempfile.mkdtemp()

# Formulario de entrada
with st.form("video_form"):
    st.subheader("Configuración del video")
    
    # Texto de introducción
    intro_text = st.text_input("Texto de introducción (máximo 10 palabras):", 
                              placeholder="Ej: Mis vacaciones de verano")
    
    # Videos
    videos = st.file_uploader("Selecciona tus videos:", 
                             type=["mp4", "mov", "avi", "mkv"], 
                             accept_multiple_files=True)
    
    # Música
    music = st.file_uploader("Selecciona tu música:", 
                            type=["mp3", "wav", "aac"])
    
    # Logo
    logo = st.file_uploader("Selecciona tu logo:", 
                           type=["jpg", "jpeg", "png"])
    
    # Botón de envío
    submitted = st.form_submit_button("Crear Video Secuencial")

# Procesamiento cuando se envía el formulario
if submitted:
    # Validar entradas
    if not intro_text:
        st.error("Por favor ingresa un texto de introducción.")
        st.stop()
    
    words = intro_text.split()
    if len(words) > 10:
        st.error("El texto de introducción debe tener máximo 10 palabras.")
        st.stop()
    
    if not videos or len(videos) == 0:
        st.error("Por favor selecciona al menos un video.")
        st.stop()
    
    if not music:
        st.error("Por favor selecciona un archivo de música.")
        st.stop()
    
    if not logo:
        st.error("Por favor selecciona un logo.")
        st.stop()
    
    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Guardar archivos subidos en el directorio temporal
    status_text.text("Guardando archivos subidos...")
    progress_bar.progress(10)
    
    video_paths = []
    for i, video in enumerate(videos):
        video_path = os.path.join(temp_dir, f"video_{i}.mp4")
        with open(video_path, "wb") as f:
            f.write(video.getbuffer())
        video_paths.append(video_path)
    
    music_path = os.path.join(temp_dir, "music.mp3")
    with open(music_path, "wb") as f:
        f.write(music.getbuffer())
    
    logo_path = os.path.join(temp_dir, "logo.png")
    with open(logo_path, "wb") as f:
        f.write(logo.getbuffer())
    
    # Crear video de introducción
    status_text.text("Creando video de introducción...")
    progress_bar.progress(20)
    
    intro_video_path = os.path.join(temp_dir, "intro.mp4")
    create_intro_video(intro_text, intro_video_path)
    
    # Crear video final con logo
    status_text.text("Creando video final con logo...")
    progress_bar.progress(30)
    
    outro_video_path = os.path.join(temp_dir, "outro.mp4")
    create_outro_video(logo_path, outro_video_path)
    
    # Concatenar videos
    status_text.text("Concatenando videos...")
    progress_bar.progress(40)
    
    all_videos = [intro_video_path] + video_paths + [outro_video_path]
    concatenated_path = os.path.join(temp_dir, "concatenated.mp4")
    concatenate_videos(all_videos, concatenated_path)
    
    # Añadir audio
    status_text.text("Añadiendo música de fondo...")
    progress_bar.progress(60)
    
    final_video_path = os.path.join(output_dir, "final_video.mp4")
    add_audio_to_video(concatenated_path, music_path, final_video_path)
    
    # Extraer audio
    status_text.text("Extrayendo audio...")
    progress_bar.progress(80)
    
    final_audio_path = os.path.join(output_dir, "final_audio.mp3")
    extract_audio(final_video_path, final_audio_path)
    
    # Completar
    progress_bar.progress(100)
    status_text.text("¡Video creado exitosamente!")
    
    # Mostrar resultado
    st.subheader("Resultado")
    st.video(final_video_path)
    
    # Botones de descarga
    col1, col2 = st.columns(2)
    
    with open(final_video_path, "rb") as f:
        video_bytes = f.read()
    
    with open(final_audio_path, "rb") as f:
        audio_bytes = f.read()
    
    col1.download_button(
        label="Descargar Video (MP4)",
        data=video_bytes,
        file_name="video_secuencial.mp4",
        mime="video/mp4"
    )
    
    col2.download_button(
        label="Descargar Audio (MP3)",
        data=audio_bytes,
        file_name="audio_secuencial.mp3",
        mime="audio/mp3"
    )
    
    # Limpiar directorios temporales
    shutil.rmtree(temp_dir)
    shutil.rmtree(output_dir)
