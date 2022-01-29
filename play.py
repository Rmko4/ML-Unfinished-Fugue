from data_io.midi_file import midi_tones_file_to_midi_file

file_0 = "postrprocessing\pure_postprocessing_one_voice.txt"
file_1 = "postrprocessing\pure_postprocessing_two_voices.txt"

midi_tones_file_to_midi_file(file_0, "one_voice.mid", modulation=8)
midi_tones_file_to_midi_file(file_1, "two_voice.mid", modulation=8)