import requests
import pandas as pd
import random
import json

def fetch_data(url):
    """Fetches data from a URL, handling connection errors."""
    try:
        response = requests.get(url, timeout=10)  # Adding a timeout
        response.raise_for_status()  # Raises an exception for HTTP error codes
        return response.json()
    except requests.exceptions.Timeout:
        print("Error: The request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def simulate_missing_data(data, missing_percentage=0.8):
    """Simulates random missing data in the user dataset with higher variability."""
    for user in data:
        # For each field, decide if it's missing based on probability
        if random.random() < missing_percentage:  # If within the missing percentage
            if random.random() < 0.33:  # 33% chance name is missing
                user['name'] = None
            if random.random() < 0.33:  # 33% chance email is missing
                user['email'] = None
            if random.random() < 0.33:  # 33% chance city is missing
                user['address']['city'] = None
            if random.random() < 0.33:  # 33% chance username is missing
                user['username'] = None
            if random.random() < 0.33:  # 33% chance phone is missing
                user['phone'] = None
            if random.random() < 0.33:  # 33% chance website is missing
                user['website'] = None
    return data

def transform_data(data):
    """Transforms the data, extracting only necessary fields and skipping rows with missing data."""
    transformed_data = []
    missing_data = []  # To track records with missing data
    
    for user in data:
        # Extract the necessary fields from the user data
        id = user.get('id', None)
        name = user.get('name', None)
        email = user.get('email', None)
        city = user.get('address', {}).get('city', None)
        username = user.get('username', None)
        phone = user.get('phone', None)
        website = user.get('website', None)

        # Check if any required field is missing
        missing_fields = []
        if name is None:
            missing_fields.append('Name')
        if email is None:
            missing_fields.append('Email')
        if city is None:
            missing_fields.append('City')
        if username is None:
            missing_fields.append('Username')
        if phone is None:
            missing_fields.append('Phone')
        if website is None:
            missing_fields.append('Website')

        if missing_fields:
            # Only include the relevant fields in the missing data printout
            missing_data.append({
                'ID': id,
                'Name': name,
                'Username': username,
                'Email': email,
                'City': city,
                'Phone': phone,
                'Website': website
            })
            # Print the missing data with the fields only
            print(f"Missing data: {{'ID': {id}, 'Name': '{name}', 'Username': '{username}', 'Email': '{email}', 'City': '{city}', 'Phone': '{phone}', 'Website': '{website}'}}")
            print(f"Missing fields: {', '.join(missing_fields)}")
            continue  # Skip users with missing data
        
        # Append the cleaned data with the necessary columns
        transformed_data.append({
            'ID': id,
            'Name': name,
            'Username': username,
            'Email': email,
            'City': city,
            'Phone': phone,
            'Website': website
        })
    
    if missing_data:
        print(f"Found {len(missing_data)} records with missing data.")
    else:
        print("No records found with missing data.")
    
    return transformed_data, missing_data

def save_to_csv(data, filename):
    """Saves the transformed data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Transformed data saved to '{filename}'.")

def save_missing_to_csv(data, filename):
    """Saves the records with missing data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Records with missing data saved to '{filename}'.")

def main():
    """Main function to control the ETL pipeline."""
    # Load the configuration
    with open("config.json", "r") as file:
        config = json.load(file)
    
    # Fetch the configuration values
    url = config.get("url")
    missing_percentage = config.get("missing_percentage", 0.1)
    transformed_data_csv = config.get("transformed_data_csv", "users_transformed.csv")
    missing_data_csv = config.get("missing_data_csv", "users_missing.csv")
    
    # Fetch the data
    data = fetch_data(url)
    if data:
        # Simulate missing data based on the given percentage
        data = simulate_missing_data(data, missing_percentage)
        
        # Transform the data
        transformed_data, missing_data = transform_data(data)
        
        # Save the transformed data
        save_to_csv(transformed_data, transformed_data_csv)
        
        # Save the records with missing data
        save_missing_to_csv(missing_data, missing_data_csv)

if __name__ == '__main__':
    main()
