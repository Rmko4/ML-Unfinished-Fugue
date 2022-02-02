from data_io.midi_file import MODULATION, TEMPO, midi_tones_to_midi_file
from data_io.model_data import convert_raw_to_training_data, load_data_raw
from postprocessing.postprocessing import PostProcessorMC
import pickle
from esn import ESN


FILE = 'models/esn/esn.pkl'
MEASURE_LEN = 16  # Lenght of a measure in symbols

RESERVOIR = 2000
W_IN = 0.3
BIAS = 0.9
SP = 1.0
LEAKING = 0.1
WASHOUT_TIME = 100
RIDGE_PARAM = 1


def apply_esn():
    midi_raw = load_data_raw('F.txt')[:-16, :]
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, window_length=1, flatten_output=True)
    u, y = data

    with open(FILE, 'rb') as inp:
        print('Loading model...')
        model = pickle.load(inp)

    # model = ESN(u.shape[1], y.shape[1], reservoir_size=RESERVOIR, W_in_scaling=W_IN,
    #             bias_scaling=BIAS, spectral_radius=SP, leaking_rate=LEAKING, ridge_param=RIDGE_PARAM,
    #             ive=ive, ove=ove, washout_time=WASHOUT_TIME, silent=False)
    # model.fit(u, y)

    post_processor = PostProcessorMC(ove, midi_raw, measure_length=MEASURE_LEN)

    u_drive = u[-300:, :]
    y_drive = y[-300:, :]
    predicted_sequence = model.predict_sequence(u_drive, y_drive, 600, post_processor)

    output_file = "output_midi_files/pred_esn_mc.mid"
    midi_tones_to_midi_file(predicted_sequence, str(output_file), tempo=TEMPO, modulation=MODULATION)

    print('Done!')


if __name__ == '__main__':
    apply_esn()
