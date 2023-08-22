import csv
import random

def generate_random_name_combinations(names, num_combinations):
    name_combinations = []

    for _ in range(num_combinations):
        random.shuffle(names)
        name_combinations.append(','.join(names))

    return name_combinations

def main():
    names_file = 'names.csv'  # Names CSV file
    ids_file = 'input.csv'  # IDs CSV file
    names_column = 0  # Index of the Names column in names CSV
    id_column = 0  # Index of the ID column in IDs CSV
    num_combinations_per_id = 5  # Number of random combinations per ID

    with open(names_file, 'r') as names_csv:
        names_reader = csv.reader(names_csv)
        names = [row[names_column] for row in names_reader]

    with open(ids_file, 'r') as ids_csv:
        ids_reader = csv.reader(ids_csv)
        next(ids_reader)  # Skip the header row if present

        id_to_names = {}

        for row in ids_reader:
            id_value = row[id_column]
            random_name_combinations = generate_random_name_combinations(names, num_combinations_per_id)
            id_to_names[id_value] = random_name_combinations

    with open('random_name_assignments.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['ID', 'Random Name Combination'])  # Write header

        for id_value, name_combinations in id_to_names.items():
            for combination in name_combinations:
                writer.writerow([id_value, combination])

if __name__ == "__main__":
    main()
