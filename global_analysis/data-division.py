import os
import tempfile
import math


def divide_dataset(file_path, output_dir, chunk_size=8 * 1024 * 1024):
    # Create the specified directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the total number of lines in the file
    total_lines = sum(1 for _ in open(file_path))

    # Calculate the approximate number of lines per chunk
    lines_per_chunk = math.ceil(chunk_size / (os.path.getsize(file_path) / total_lines))

    # Initialize variables
    file_counter = 1
    chunk_counter = 1
    line_counter = 0

    # Open the large dataset file
    with open(file_path, 'r') as file:
        # Read the header
        header = file.readline()

        while True:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                # Write the header to the temporary file
                temp_file.write(header)

                # Write the chunk to the temporary file
                for line in file:
                    temp_file.write(line)
                    line_counter += 1

                    if line_counter >= lines_per_chunk:
                        line_counter = 0
                        break

            # Create a new file name
            output_file = f"{output_dir}/file_{file_counter}.csv"

            # Rename the temporary file to the output file
            os.rename(temp_file.name, output_file)

            print(f"Chunk {chunk_counter} written to {output_file}")

            # Increment the counters
            file_counter += 1
            chunk_counter += 1

            # Check if the end of the file is reached
            if not line:
                break

    print("Dataset division completed.")


# Specify the path to your large dataset file
large_dataset_file = '/Users/GoldenEagle/Downloads/BTCUSDT-trades-2024-03-08.csv'

# Specify the directory where you want to save the chunk files
output_directory = '/Users/GoldenEagle/PycharmProjects/ML_models-infrastructure/global_analysis/chunk'

# Call the function to divide the dataset
divide_dataset(large_dataset_file, output_directory)