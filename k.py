import webbrowser
import time

import csv

def import_csv(filename):
    links = []
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the second column contains the desired values
            if len(row) >= 2:
                links.append(row[3])
    
    return links

# Usage example
filename = 'example3.csv'  # Replace with your actual CSV file name


# Infinite loop to continuously open links
while True:
    links = import_csv(filename)
    for link in links:
        try:
            webbrowser.open(link)
            # You can add any additional actions or delays between opening links here
            time.sleep(2)  # Wait for 2 seconds before opening the next link
            
        except Exception as e:
            # Handle any exceptions that may occur during the process
            print(f"Encountered an error while processing link: {link}")
            print(f"Error message: {str(e)}")
    
    # Once all links have been processed, start over from the beginning of the list

# The infinite loop will continue indefinitely until manually interrupted

