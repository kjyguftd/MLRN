import os
import re
from pathlib import Path

from cleantext import clean
from datetime import datetime
import pandas as pd


def filter_data(raw_data):
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    example_date = datetime.strptime(raw_data['datetime'], '%Y-%m-%d')

    return (raw_data['dataType'] == 'comment') and (start_date <= example_date <= end_date)


def write_to_csv_by_label(dataset, split_name):
    """
    :param dataset
    :param split_name
    """
    df = pd.DataFrame(dataset)
    labels = df['label'].unique()

    for label in labels:
        label_df = df[df['label'] == label]
        directory = os.path.join(f"dataset/{split_name}")
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{label.replace('/', '_')}.csv")
        label_df.to_csv(file_path, escapechar='\\', index=False)


def clean_comment(text):
    """
    :param text: raw data
    :return: cleaned data
    """
    cleaned_text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_numbers=True,
        no_digits=True,
        no_currency_symbols=True,
        no_punct=True,
        replace_with_punct="",
        replace_with_url="<URL>",
        replace_with_number="<NUMBER>",
        replace_with_digit="",
        replace_with_currency_symbol="<CUR>",
        lang='en'
    )

    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
    return cleaned_text


def write_cleaned_data(source_folder, destination_folder):
    """
    :param source_folder
    :param destination_folder
    """
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        print(source_file, destination_file)
        filtered_data = pd.read_csv(source_file, encoding='utf-8')
        filtered_data['cleaned_text'] = filtered_data['text'].apply(clean_comment)
        filtered_data.pop('text')
        filtered_data.insert(0, 'cleaned_text', filtered_data.pop('cleaned_text'))
        filtered_data.to_csv(destination_file, index=False)