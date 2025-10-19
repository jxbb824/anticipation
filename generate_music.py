# from transformers import AutoModelForCausalLM
# import os
# import zipfile
# import time
# from anticipation.sample import generate
# from anticipation.convert import events_to_midi

# def generate_music(model_path, model_name, count=10, length=10):
#     """
#     Generate multiple music pieces from the specified model
    
#     Args:
#         model_path: Model path or name
#         model_name: Model identifier for file naming
#         count: Number of music pieces to generate
#         length: Length of each music piece (seconds)
    
#     Returns:
#         List of generated MIDI file paths
#     """
#     print(f"Loading model {model_path}...")
#     model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    
#     midi_files = []
#     model_dir = f"generated_music/{model_name}"
#     os.makedirs(model_dir, exist_ok=True)
    
#     for i in range(count):
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         file_name = f"{model_dir}/{model_name}_piece_{i+1}_{timestamp}.mid"
        
#         print(f"Generating piece {i+1}/{count} for {model_name}...")
#         events = generate(model, start_time=0, end_time=length, top_p=.98, debug=False)
#         mid = events_to_midi(events)
#         mid.save(file_name)
#         midi_files.append(file_name)
#         time.sleep(1)  # Ensure different timestamps
    
#     return midi_files

# # Create main output directory
# os.makedirs("generated_music", exist_ok=True)

# # Generate music from the first model (stanford-crfm/music-small-800k)
# model1_path = 'stanford-crfm/music-small-800k'
# model1_name = "stanford_model"
# model1_files = generate_music(model1_path, model1_name)

# # Generate music from the second model (finetune_output/final_model)
# model2_path = '/home/xiruij/anticipation/finetune_output/final_model'
# model2_name = "finetune_model"
# model2_files = generate_music(model2_path, model2_name)

# # Create a ZIP archive containing all MIDI files
# zip_filename = "generated_music/all_music.zip"
# with zipfile.ZipFile(zip_filename, 'w') as zipf:
#     for file in model1_files + model2_files:
#         # Keep directory structure but remove "generated_music/" prefix
#         arcname = file[len("generated_music/"):]
#         zipf.write(file, arcname=arcname)

# print(f"All music has been generated and saved to {zip_filename}")



from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-small-800k').cuda()
# model = AutoModelForCausalLM.from_pretrained('/home/xiruij/anticipation/finetune_subset_output/final_model')
model = AutoModelForCausalLM.from_pretrained('/home/xiruij/anticipation/checkpoints_subset_large/full_model')
# print(vars(model)['_modules'])
from anticipation.sample import generate
from anticipation.convert import events_to_midi

length = 15 # time in seconds
events = generate(model, start_time=0, end_time=length, top_p=.98, debug=True)
mid = events_to_midi(events)
mid.save('tmp.mid')
