# Project_2: ETL Pipeline for User Data

This project implements an ETL (Extract, Transform, Load) pipeline that fetches user data from an external API, simulates missing data, transforms the data, and saves the transformed data and records with missing fields into CSV files.

## Features

- Fetches user data from a public API (`https://jsonplaceholder.typicode.com/users`).
- Simulates missing data in user records (name, email, city, username, phone, website).
- Extracts necessary fields from the data and identifies records with missing information.
- Saves the transformed data and the records with missing data into separate CSV files.

## Setup

1. **Clone the repository**:
   If you haven't already, clone the repository to your local machine.

   ```bash
   git clone https://github.com/PepeLavadoRalo/projects.git
   ```
   
2. **Set up the virtual environment**: Set up the virtual environment: Create and activate a Python virtual environment.
  ```bash
  python -m venv venv
  # On Windows
  venv\Scripts\activate
  # On macOS/Linux
  source venv/bin/activate
  ```
3. **Install dependencies**: Install the required packages using pip.
  ```bash
   pip install -r requirements.txt
   ```

## Configuration

The project uses a config.json file to store configuration settings. The default configuration is:
```json
{
  "url": "https://jsonplaceholder.typicode.com/users",
  "missing_percentage": 0.1,
  "transformed_data_csv": "users_transformed.csv",
  "missing_data_csv": "users_missing.csv"
}
```

You can change the values of the configuration parameters:

- `url`: The API endpoint to fetch the data from (default is `https://jsonplaceholder.typicode.com/users`).
- `missing_percentage`: The percentage of missing data to simulate (default is `0.1`).
- `transformed_data_csv`: The filename for saving the transformed data (default is `users_transformed.csv`).
- `missing_data_csv`: The filename for saving the records with missing data (default is `users_missing.csv`).

## How to Run

Once you've set up your environment, simply run the script:

```bash
python pipeline_etl.py
```
## The Script Will:

- Fetch data from the configured API.
- Simulate missing data based on the `missing_percentage`.
- Transform the data by removing records with missing fields.
- Save the transformed data and missing data into CSV files (`users_transformed.csv` and `users_missing.csv`).

## Example Output

The script will output missing data information to the console, for example:

```kotlin
Missing data: {'ID': 9, 'Name': 'Glenna Reichert', 'Username': 'Delphine', 'Email': 'Chaim_McDermott@dana.io', 'City': None, 'Phone': '(775)976-6794 x41206', 'Website': None}
Missing fields: City, Website
Found 1 records with missing data.
Transformed data saved to 'users_transformed.csv'.
Records with missing data saved to 'users_missing.csv'.
```

## Files

- `pipeline_etl.py`: The main Python script that performs the ETL process.
- `config.json`: The configuration file containing API URL and file names.
- `users_transformed.csv`: CSV file containing the cleaned data.
- `users_missing.csv`: CSV file containing the records with missing fields.
- `requirements.txt`: List of Python dependencies.



