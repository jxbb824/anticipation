from anticipation.convert import events_to_midi
file_path = '/home/xiruij/anticipation/datasets/clap/train_unique.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()
    fifth_line = lines[2419].strip()

events = fifth_line.split()
events = events[1:-1]
events = [int(event) for event in events]
mid = events_to_midi(events)
mid.save('generated.mid')