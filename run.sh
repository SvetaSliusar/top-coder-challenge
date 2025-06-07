#!/bin/bash

trip_duration_days=$1
miles_traveled=$2
total_receipts_amount=$3

# Call Python process.py in predict mode
python3 process_data.py predict "$trip_duration_days" "$miles_traveled" "$total_receipts_amount"
