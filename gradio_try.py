import whisper
from got import Got
import os
import gradio as gr

# Load Whisper model
whisper_model = whisper.load_model("base")

# Step 1: Transcribe audio
def transcribe_audio(audio_path):
    """
    Transcribe audio using Whisper.
    """
    result = whisper_model.transcribe(audio_path, fp16=False)
    return result["text"]

# Step 2: Analyze transcription with Got
def analyze_transcription(transcription):
    """
    Analyze the transcription using Got.
    """
    prompt = f"""
    The following is a conversation between a doctor and a patient:
    {transcription}

    Based on this conversation, provide:
    1. A possible prognosis for the patient.
    2. A detailed diagnosis of the condition.
    3. Medication recommendations or treatments for the patient.
    """
    client = Got(api_key=os.getenv("GOT_API_KEY"))
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a medical assistant AI with expertise in prognosis, diagnosis, and medication recommendations."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Full Gradio Workflow
def process_audio(file):
    """
    Full workflow for processing audio and providing medical analysis.
    """
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(file.read())

    # Transcription
    transcription = transcribe_audio("temp_audio.wav")

    # Analysis
    analysis = analyze_transcription(transcription)

    return transcription, analysis

# Gradio Interface
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(source="upload", type="file"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Medical Analysis")
    ],
    title="Medical Audio Analysis App",
    description="Upload an audio file of a doctor-patient conversation to get transcription and medical recommendations."
)

if __name__ == "__main__":
    interface.launch()
