import gigaam

audio_path = "test_audio.ogg"

model_name = "v3_ssl"  
model = gigaam.load_model(model_name)

embedding, _ = model.embed_audio(audio_path)

print(embedding)

model_name = "v3_e2e_rnnt" 

model = gigaam.load_model(model_name)
transcription = model.transcribe(audio_path)
print(transcription)

model = gigaam.load_model("emo")
emotion2prob = model.get_probs(audio_path)
print(", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in emotion2prob.items()]))