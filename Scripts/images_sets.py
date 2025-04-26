import os
import numpy as np
import pickle

IMAGES_PATH = "/Users/pedronicolas/Desktop/jjk_classifier/image_dataset2"
PKL_PATH = "/Users/pedronicolas/Desktop/jjk_classifier/dataset_pkl_complete2.pkl"
NPZ_PATH = "/Users/pedronicolas/Desktop/jjk_classifier/dataset_npz_complete2.npz"

characters = ["Gojo", "Hakari", "Megumi","Sukuna","Yuji","Yuta"]
num_sequences = 30
sequence_length = 30
label_map = {label:num for num,label in enumerate(characters)}

sequences,labels = [] , []
for character in characters:
  for sequence in range(num_sequences):
    window = []
    for frame_num in range(sequence_length):
      res = np.load(os.path.join(IMAGES_PATH,character,str(sequence),"{}.npy".format(frame_num)))
      window.append(res)
    sequences.append(window)
    labels.append(label_map[character])


# Saving .pkl (listas)
with open(PKL_PATH, "wb") as f:
    pickle.dump((sequences, labels), f)
print(f"Dataset saved in pkl: {PKL_PATH}")

# Convert to numpy arrays
X = np.array(sequences)
y = np.array(labels)

# Saving .npz
np.savez_compressed(NPZ_PATH, X=X, y=y)
print(f"Dataset saved in .npz: {NPZ_PATH}")
print("final shapes For: X =", X.shape, "y =", y.shape)