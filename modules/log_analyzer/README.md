# Log Analyzer for Integration Data

This script analyzes log data to identify cases where the time difference between processing time and registration time exceeds a threshold (default: 10 seconds).

## Features

- Parses JSON log data from CSV files
- Extracts only the filename from image URLs
- Calculates time differences between processing and registration timestamps
- Groups results by date
- Creates separate output files for each date
- Counts the number of lesions in each record
- Generates a summary report with statistics for each date
- Creates a dedicated output folder based on the input file name

## Usage

```bash
python3 analyze_time_difference.py [input_file_path]
```

If no input file path is provided, the script will use the default path: `/home/diego/db_industry_integration_log_prn.csv`

## Output

The script creates a folder with the same name as the input file (without extension) and saves the following files inside:

1. `time_difference_results_YYYY-MM-DD.csv` - One file for each date found in the logs, containing:
   - Image ID
   - Image filename
   - Processing time
   - Registration time
   - Time difference in seconds
   - Number of lesions

2. `time_difference_summary.csv` - A summary file with statistics for each date:
   - Date
   - Total entries (records above the time threshold)
   - Total records (all records for that date)
   - Percentage (percentage of records above the threshold)
   - Average time difference
   - Maximum time difference

## Example

```bash
python3 analyze_time_difference.py /path/to/your/logfile.csv
```

This will create a folder named `logfile` containing the analysis results.
