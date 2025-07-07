#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to analyze log data and identify cases where the time difference between
data_hora_hora_processamento and data_hora_registro is greater than 10 seconds.
Creates separate output files for each day found in the log data.
"""

import json
import csv
import datetime
from datetime import datetime as dt
import sys
import os
import re
from collections import defaultdict


def parse_datetime(datetime_str):
    """Parse datetime string to datetime object."""
    if '.' in datetime_str:
        # Format with microseconds
        return dt.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    else:
        # Format without microseconds
        return dt.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")


def calculate_time_difference(processing_time, registration_time):
    """Calculate time difference in seconds between two datetime strings."""
    processing_dt = parse_datetime(processing_time)
    registration_dt = parse_datetime(registration_time)
    
    # Calculate the difference in seconds
    diff = (processing_dt - registration_dt).total_seconds()
    return diff


def extract_filename(image_url):
    """Extract just the filename from the image URL."""
    # Extract the filename using regex or string operations
    match = re.search(r'(\d{8}-\d{4}-\d-\d{4}-\d{4}\.jpg)', image_url)
    if match:
        return match.group(1)
    else:
        # Fallback to basic path extraction if regex doesn't match
        return os.path.basename(image_url)


def extract_date(datetime_str):
    """Extract the date part from a datetime string."""
    return datetime_str.split()[0]


def create_output_directory(input_file):
    """
    Create an output directory based on the input file name.
    
    Args:
        input_file (str): Path to the input CSV file
    
    Returns:
        str: Path to the output directory
    """
    # Get the base name of the input file without extension
    base_name = os.path.basename(input_file)
    file_name_without_ext = os.path.splitext(base_name)[0]
    
    # Create output directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, file_name_without_ext)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir


def write_summary_file(summary_data, output_dir):
    """
    Write a summary file with totals for each date.
    
    Args:
        summary_data (dict): Dictionary containing summary data
        output_dir (str): Directory to save the summary file
    """
    summary_file = os.path.join(output_dir, "time_difference_summary.csv")
    
    with open(summary_file, 'w', newline='') as csvfile:
        fieldnames = ['date', 'total_entries', 'total_records', 'percentage', 'avg_time_difference', 'max_time_difference']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for date, data in sorted(summary_data.items()):
            # Calculate percentage of records above threshold
            percentage = 0
            if data['total_records'] > 0:
                percentage = (data['total_entries'] / data['total_records']) * 100
            
            writer.writerow({
                'date': date,
                'total_entries': data['total_entries'],
                'total_records': data['total_records'],
                'percentage': round(percentage, 2),
                'avg_time_difference': round(data['avg_time_difference'], 2),
                'max_time_difference': round(data['max_time_difference'], 2)
            })
    
    print(f"Summary report saved to {summary_file}")


def analyze_log_file(input_file, output_dir, threshold_seconds=10):
    """
    Analyze log file to find entries where time difference exceeds threshold.
    Creates separate output files for each day found in the log data.
    
    Args:
        input_file (str): Path to the input CSV file containing log data
        output_dir (str): Directory to save output CSV files
        threshold_seconds (int): Threshold in seconds for time difference
    """
    # Group results by date
    results_by_date = defaultdict(list)
    
    # Keep track of total records per date
    total_records_by_date = defaultdict(int)
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    # The line is a CSV field containing a JSON string with escaped quotes
                    # First, remove the outer quotes if they exist
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    
                    # Replace double escaped quotes with single quotes
                    line = line.replace('""', '"')
                    
                    # Parse the JSON data
                    data = json.loads(line)
                    
                    # Some lines might contain a list with a single item
                    if isinstance(data, list):
                        data = data[0]
                    
                    # Extract the required fields
                    id_imagem = data.get('id_imagem')
                    imagem_url = data.get('imagem')
                    processing_time = data.get('data_hora_hora_processamento')
                    registration_time = data.get('data_hora_registro')
                    
                    if processing_time:
                        # Extract date from processing time for grouping
                        date = extract_date(processing_time)
                        # Increment total records counter for this date
                        total_records_by_date[date] += 1
                    
                    # Extract the number of lesions
                    num_lesoes = 0
                    if 'dados' in data and 'lesoes' in data['dados']:
                        num_lesoes = len(data['dados']['lesoes'])
                    
                    if processing_time and registration_time:
                        # Calculate time difference
                        time_diff = calculate_time_difference(processing_time, registration_time)
                        
                        # Check if time difference exceeds threshold
                        if time_diff > threshold_seconds:
                            # Extract just the filename from the image URL
                            imagem_filename = extract_filename(imagem_url)
                            
                            # Extract date from processing time for grouping
                            date = extract_date(processing_time)
                            
                            results_by_date[date].append({
                                'id_imagem': id_imagem,
                                'imagem': imagem_filename,
                                'data_hora_hora_processamento': processing_time,
                                'data_hora_registro': registration_time,
                                'time_difference_seconds': time_diff,
                                'num_lesoes': num_lesoes
                            })
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print(f"Problematic line: {line[:50]}...")
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
        
        # Write results to separate CSV files for each date and collect summary data
        total_entries = 0
        summary_data = {}
        
        for date, results in results_by_date.items():
            if results:
                output_file = os.path.join(output_dir, f"time_difference_results_{date}.csv")
                
                with open(output_file, 'w', newline='') as csvfile:
                    fieldnames = ['id_imagem', 'imagem', 'data_hora_hora_processamento', 'data_hora_registro', 'time_difference_seconds', 'num_lesoes']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for result in results:
                        writer.writerow(result)
                
                # Collect summary data for this date
                time_diffs = [float(result['time_difference_seconds']) for result in results]
                
                summary_data[date] = {
                    'total_entries': len(results),
                    'total_records': total_records_by_date[date],
                    'avg_time_difference': sum(time_diffs) / len(time_diffs) if time_diffs else 0,
                    'max_time_difference': max(time_diffs) if time_diffs else 0
                }
                
                print(f"Found {len(results)} entries for date {date} with time difference > {threshold_seconds} seconds.")
                print(f"Total records for date {date}: {total_records_by_date[date]}")
                print(f"Results saved to {output_file}")
                total_entries += len(results)
        
        # Add dates that have records but no entries above threshold
        for date, count in total_records_by_date.items():
            if date not in summary_data:
                summary_data[date] = {
                    'total_entries': 0,
                    'total_records': count,
                    'avg_time_difference': 0,
                    'max_time_difference': 0
                }
        
        if total_entries > 0:
            print(f"Analysis complete. Found a total of {total_entries} entries with time difference > {threshold_seconds} seconds.")
            print(f"Total records processed: {sum(total_records_by_date.values())}")
            
            # Generate summary file
            write_summary_file(summary_data, output_dir)
        else:
            print(f"No entries found with time difference > {threshold_seconds} seconds.")
            print(f"Total records processed: {sum(total_records_by_date.values())}")
            
            # Generate summary file even if no entries above threshold
            if total_records_by_date:
                write_summary_file(summary_data, output_dir)
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


def main():
    """Main function to run the script."""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "/home/diego/Desktop/db_industry_integration_log_rlm_big.csv"
    
    # Create output directory based on input file name
    output_dir = create_output_directory(input_file)
    
    # Run analysis
    analyze_log_file(input_file, output_dir, threshold_seconds=10)


if __name__ == "__main__":
    main()
