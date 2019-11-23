import os
from datetime import datetime
import numpy as np


def create_csv(results, results_dir='./'):
    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:
        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


def predict_submissions(model, test_dataset, test_gen, result_dir="./", ):
    predictions = model.predict(x=test_dataset, steps=len(test_gen), verbose=1)
    predicted_class = np.argmax(predictions, axis=1)

    image_names = [filename.split('\\')[-1] for filename in test_gen.filenames]


    results = dict(zip(image_names, predicted_class))
    print(results)
    create_csv(results, results_dir=result_dir)

    print("Wrote file .csv")
