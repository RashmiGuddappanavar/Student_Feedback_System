import csv
import pandas as pd
import numpy as np

# Function to write department feedback data to CSV
def write_to_csv_departments(time, teachingscore, teaching, placementsscore, placements,
                              collaboration_with_companiesscore, collaboration_with_companies,
                              infrastructurescore, infrastructure, hostelscore, hostel,
                              libraryscore, library):
    try:
        with open('dataset/database.csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

        with open('dataset/database.csv', "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            record = {
                'Timestamp': time,
                'teachingscore': teachingscore,
                'teaching': teaching,
                'placementsscore': placementsscore,
                'placements': placements,
                'collaboration_with_companiesscore': collaboration_with_companiesscore,
                'collaboration_with_companies': collaboration_with_companies,
                'infrastructurescore': infrastructurescore,
                'infrastructure': infrastructure,
                'hostelscore': hostelscore,
                'hostel': hostel,
                'libraryscore': libraryscore,
                'library': library
            }
            writer.writerow(record)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Function to write teacher feedback data to CSV
def write_to_csv_teachers(teacher1, teacher1score, teacher2, teacher2score, teacher3, teacher3score,
                          teacher4, teacher4score, teacher5, teacher5score, teacher6, teacher6score):
    try:
        with open('dataset/teacherdb.csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

        with open('dataset/teacherdb.csv', "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            record = {
                'teacher1': teacher1, 'teacher1score': teacher1score,
                'teacher2': teacher2, 'teacher2score': teacher2score,
                'teacher3': teacher3, 'teacher3score': teacher3score,
                'teacher4': teacher4, 'teacher4score': teacher4score,
                'teacher5': teacher5, 'teacher5score': teacher5score,
                'teacher6': teacher6, 'teacher6score': teacher6score
            }
            writer.writerow(record)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Function to get the feedback counts from the dataset
def get_counts():
    try:
        path = 'dataset/database.csv'
        df = pd.read_csv(path)

        required_columns = [
            'teachingscore', 'placementsscore', 'collaboration_with_companiesscore',
            'infrastructurescore', 'hostelscore', 'libraryscore'
        ]

        for col in required_columns:
            if col not in df.columns or df[col].isnull().all():
                df[col] = 0

        index = df.index
        no_of_students = len(index)
        total_feedbacks = no_of_students * 7

        def count_feedbacks(score_column):
            grouped = df[score_column].value_counts()
            negative = grouped.get(-1, 0)
            neutral = grouped.get(0, 0)
            positive = grouped.get(1, 0)
            return negative, neutral, positive

        teaching_counts = count_feedbacks('teachingscore')
        placements_counts = count_feedbacks('placementsscore')
        collaboration_counts = count_feedbacks('collaboration_with_companiesscore')
        infrastructure_counts = count_feedbacks('infrastructurescore')
        hostel_counts = count_feedbacks('hostelscore')
        library_counts = count_feedbacks('libraryscore')

        total_positive_feedbacks = sum(count[2] for count in [
            teaching_counts, placements_counts, collaboration_counts,
            infrastructure_counts, hostel_counts, library_counts
        ])
        total_neutral_feedbacks = sum(count[1] for count in [
            teaching_counts, placements_counts, collaboration_counts,
            infrastructure_counts, hostel_counts, library_counts
        ])
        total_negative_feedbacks = sum(count[0] for count in [
            teaching_counts, placements_counts, collaboration_counts,
            infrastructure_counts, hostel_counts, library_counts
        ])

        if total_feedbacks == 0:
            return no_of_students, 0, 0, 0, []

        raw_positive = total_positive_feedbacks / total_feedbacks * 100
        raw_neutral = total_neutral_feedbacks / total_feedbacks * 100
        raw_negative = total_negative_feedbacks / total_feedbacks * 100

        positive_percentage = round(raw_positive)
        neutral_percentage = round(raw_neutral)
        negative_percentage = round(raw_negative)

        total_percentage = positive_percentage + neutral_percentage + negative_percentage
        difference = 100 - total_percentage

        if difference != 0:
            if abs(raw_positive - positive_percentage) >= abs(raw_neutral - neutral_percentage) and \
               abs(raw_positive - positive_percentage) >= abs(raw_negative - negative_percentage):
                positive_percentage += difference
            elif abs(raw_neutral - neutral_percentage) >= abs(raw_negative - negative_percentage):
                neutral_percentage += difference
            else:
                negative_percentage += difference

        counts = [
            *teaching_counts, *placements_counts, *collaboration_counts,
            *infrastructure_counts, *hostel_counts, *library_counts
        ]

        return no_of_students, positive_percentage, negative_percentage, neutral_percentage, counts

    except Exception as e:
        print(f"Error reading dataset: {e}")
        return 0, 0, 0, 0, []

# Function to get the latest tables from the dataset
def get_tables():
    try:
        df = pd.read_csv('dataset/database.csv')
        df = df.tail(5)
        return [df.to_html(classes='data')]
    except Exception as e:
        print(f"Error reading tables: {e}")
        return []

# Function to get the column titles of the dataset
def get_titles():
    try:
        df = pd.read_csv('dataset/database.csv')
        return df.columns.values
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return []

# Example Usage
if __name__ == "__main__":
    no_of_students, positive_percentage, negative_percentage, neutral_percentage, counts = get_counts()
    print("Number of Students:", no_of_students)
    print("Positive Feedback Percentage:", positive_percentage)
    print("Neutral Feedback Percentage:", neutral_percentage)
    print("Negative Feedback Percentage:", negative_percentage)
    print("Counts by Aspect:", counts)
