import numpy as np
from tensorflow.keras.models import load_model

# Define character mapping
EON_CHAR = '`'
chars = "abcdefghijklmnopqrstuvwxyz`"  # Add all characters including EON_CHAR
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Load the trained model
model_filename = '0702-650546223-POPRYHO.ZZZ'
model = load_model(model_filename)

# Name generation function
def generate_name(model, start_char='a', name_length=11, temperature=1.0):
    name = start_char
    seq_length = 11
    while len(name) < name_length and name[-1] != EON_CHAR:
        pattern = [char_to_int[char] for char in name] + [char_to_int[EON_CHAR]] * (seq_length - len(name))
        one_hot_pattern = np.zeros((1, seq_length, len(chars)), dtype=bool)
        for i, char_index in enumerate(pattern):
            one_hot_pattern[0, i, char_index] = 1
        prediction = model.predict(one_hot_pattern, verbose=0)[0][-1]
        prediction = np.log(prediction + 1e-7) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(chars), p=prediction)
        next_char = int_to_char[next_index]
        if next_char == EON_CHAR:
            break
        name += next_char
    return name.replace(EON_CHAR, '')

# Generate names starting with 'a' and 'x'
generated_names_a = [generate_name(model, start_char='a') for _ in range(20)]
generated_names_x = [generate_name(model, start_char='x') for _ in range(20)]

print("Names starting with 'a':", generated_names_a)
print("Names starting with 'x':", generated_names_x)
