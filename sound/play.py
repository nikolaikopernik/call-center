from mlx_audio.tts.generate import generate_audio
import time

start = time.time()
generate_audio(text = "Wilders schrijft op X dat de Nationaal Coördinator Terrorismebestrijding en Veiligheid (NCTV) aan hem heeft bevestigd dat hij door de gearresteerde verdachten van de verijdelde aanslag op De Wever ook is genoemd als doelwit. \"De NCTV verwacht geen 'restdreiging', maar ik heb hier zelf een slecht gevoel bij en dus schort ik al mijn campagneactiviteiten voorlopig op.\"",
               model_path="mlx-community/Spark-TTS-0.5B-fp16",
               lang_code="nl",
               play=True)
print(f"Done {time.time() - start}")
