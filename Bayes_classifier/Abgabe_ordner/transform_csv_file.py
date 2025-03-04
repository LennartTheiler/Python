# In this file we will transform the dataset. The goal is to create a new csv file with the same date, except that there
# are no yes and no values. Instead we will use 1 and 0. This is necessary for the bayes classifier.
# Problem: The dataset is not in the correct format. The values are not numerical. We need to transform the dataset.
# Problem 2: The dataset is encoded in utf-16.
# Note: UTF-16 uses 2 to 4 bytes to encode a character. UTF-8 uses 1 to 4 bytes to encode a character.
# Solution: We will use the pandas library to read the csv file. We will set the encoding to utf-16. We will also set the
# delimiter to '\t' because the values are separated by tabs. We will replace the values 'yes' and 'no' with 1 and 0.
import pandas as pd

# This function reads a csv file, transforms it and saves it as a new csv file.
def transform_csv_file(input_path, output_path):

    try:
        # CSV-File is read with the pandas library. The encoding is set to utf-16 and the delimiter to '\t'.
        data = pd.read_csv(input_path, encoding='utf-16', delimiter='\t', header=None)

        # The columns are renamed.
        data.columns = ['temperature', 'nausea', 'lumbar_pain', 'urine_push', 'bladder_pain', 'urethral_discomfort',
                        'urinary_bladder_infl', 'kidney_infl']
        # Convert temperature to float
        data['temperature'] = data['temperature'].str.replace(',', '.').astype(float)

        # This option is set to avoid silent downcasting. This means that the data types of the columns are not changed
        # automatically. Example: float64 will not be downcasted to float32.
        pd.set_option('no_silent_downcasting', True)

        # The values 'yes' and 'no' are replaced with 1 and 0.
        data = data.replace({'no': 0, 'yes': 1})

        # The transformed dataset is saved as a new csv file.
        data.to_csv(output_path, index=False)
    except FileNotFoundError:
        print(f'Error: File not found at {input_path}. Please check the file path.')
    except ValueError as e:
        print(f"Error: {e}")
        print("Unexpected Values. Please check the CSV file.")
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

if __name__ == '__main__':
    input_path = 'acute_inflammations.csv'
    output_path = 'acute_inflammations_transformed.csv'
    transform_csv_file(input_path, output_path)
