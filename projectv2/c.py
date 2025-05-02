import pandas as pd
from collections import defaultdict
import csv
import re
from flask import Flask

def analyze_covid_deaths():
    """Analyze COVID deaths by year, region, age, and race from the CSV file and return a DataFrame"""
    records = []

    with open('Provisional_COVID-19_death_counts_and_rates_by_month__jurisdiction_of_residence__and_demographic_characteristics_20250415.csv', 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            if row['jurisdiction_residence'] == "United States":
                continue

            match = re.match(r"Region (\d+)", row['jurisdiction_residence'])
            if not match:
                continue

            year = row['year']
            region = int(match.group(1))
            group = row['group']
            deaths = row['COVID_deaths']
            race = row['subgroup1']
            age = row['subgroup2']

            if group != "Race and Age":
                continue

            if not deaths or not deaths.isdigit():
                continue

            # Add a row to the dataset
            records.append({
                'Year': int(year),
                'Region': region,
                'Age Group': age,
                'Race': race,
                'COVID Deaths': int(deaths)
            })

    df = pd.DataFrame(records)
    return df

# Define HHS regions globally
hhs_regions = {
        '1': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont'],
        '2': ['New Jersey', 'New York'],
        '3': ['Delaware', 'District of Columbia', 'Maryland', 'Pennsylvania', 'Virginia', 'West Virginia'],
        '4': ['Alabama', 'Florida', 'Georgia', 'Kentucky', 'Mississippi', 'North Carolina', 'South Carolina', 'Tennessee'],
        '5': ['Illinois', 'Indiana', 'Michigan', 'Minnesota', 'Ohio', 'Wisconsin'],
        '6': ['Arkansas', 'Louisiana', 'New Mexico', 'Oklahoma', 'Texas'],
        '7': ['Iowa', 'Kansas', 'Missouri', 'Nebraska'],
        '8': ['Colorado', 'Montana', 'North Dakota', 'South Dakota', 'Utah', 'Wyoming'],
        '9': ['Arizona', 'California', 'Hawaii', 'Nevada'],
        '10': ['Alaska', 'Idaho', 'Oregon', 'Washington']
}

def analyze_hhs_regions():
    """Build a DataFrame of total population per HHS region per year (2020–2025)"""
    records = []

    with open('NST-EST2024-ALLDATA.csv', 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            if row['SUMLEV'] == '040' and row['NAME'] != 'Puerto Rico':
                state_name = row['NAME']

                # Determine region number for the state
                for region, states in hhs_regions.items():
                    if state_name in states:
                        for year in ['2020', '2021', '2022', '2023', '2024']:
                            pop = int(row[f'POPESTIMATE{year}'])
                            records.append({
                                'Region': int(region),
                                'Year': int(year),
                                'Population': pop
                            })
                        # Include the 2025 base estimate as "2025"
                        records.append({
                            'Region': int(region),
                            'Year': 2025,
                            'Population': int(row['ESTIMATESBASE2020'])
                        })
                        break

    df = pd.DataFrame(records)

    # Aggregate to get total population per region per year
    df_grouped = df.groupby(['Region', 'Year'], as_index=False).agg({'Population': 'sum'})

    return df_grouped

def merge_pop(covid_results, hhs_results, political_results):
    """Calculate COVID deaths as percentage of population for each region and year"""
    covid_results['Region'] = covid_results['Region'].astype(int)
    hhs_results['Region'] = hhs_results['Region'].astype(int)
    political_results['Region'] = political_results['Region'].astype(int)
    # Merge COVID data with population data on Region and Year
    merged = pd.merge(covid_results, hhs_results, on=['Region', 'Year'], how='left')
    merged = pd.merge(merged, political_results, on=['Region', 'Year'], how='left')
    column_order = ['Year', 'Region', 'Age Group', 'Race', 'Population', 
                    'Legislature Republican %', 'Legislature Democrat %', 'Legislature Mixed %', 
                    'Governor Republican %', 'Governor Democrat %', 'State Control Republican %', 
                    'State Control Democrat %', 'State Control Mixed %','COVID Deaths']
    merged = merged[column_order]
    return merged
    

def analyze_political_control():
    """Analyze political control percentages by region and year"""
    # Load political control data
    political_df = pd.read_csv('state_political_control_2020_2025.csv')

    # Dictionary mapping states to HHS regions
    state_to_region = {}
    for region, states in hhs_regions.items():
        for state in states:
            state_to_region[state] = region

    # Prepare a list to collect rows for the DataFrame
    results = []
    years = sorted(political_df['Year'].unique())

    for year in years:
        year_data = political_df[political_df['Year'] == year]
        region_stats = defaultdict(lambda: defaultdict(int))
        region_totals = defaultdict(int)

        for _, row in year_data.iterrows():
            if row['State'] in state_to_region:
                region = state_to_region[row['State']]
                region_totals[region] += 1

                # Legislature control
                if row['Legislature_Control'] == 2:
                    region_stats[region]['rep_leg'] += 1
                elif row['Legislature_Control'] == 1:
                    region_stats[region]['dem_leg'] += 1
                else:
                    region_stats[region]['mix_leg'] += 1

                # Governor control
                if row['Governor_Control'] == 2:
                    region_stats[region]['rep_gov'] += 1
                elif row['Governor_Control'] == 1:
                    region_stats[region]['dem_gov'] += 1

                # State control
                if row['State_Control'] == 2:
                    region_stats[region]['rep_state'] += 1
                elif row['State_Control'] == 1:
                    region_stats[region]['dem_state'] += 1
                else:
                    region_stats[region]['mix_state'] += 1

        # Prepare data for DataFrame
        for region in sorted(region_totals.keys(), key=int):
            total = region_totals[region]
            stats = region_stats[region]
            results.append({
                'Year': year,
                'Region': region,
                'Legislature Republican %': stats['rep_leg'] / total * 100,
                'Legislature Democrat %': stats['dem_leg'] / total * 100,
                'Legislature Mixed %': stats['mix_leg'] / total * 100,
                'Governor Republican %': stats['rep_gov'] / total * 100,
                'Governor Democrat %': stats['dem_gov'] / total * 100,
                'State Control Republican %': stats['rep_state'] / total * 100,
                'State Control Democrat %': stats['dem_state'] / total * 100,
                'State Control Mixed %': stats['mix_state'] / total * 100
            })

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    return df


#Main function that runs when the script is executed directly
covid_results = analyze_covid_deaths()
hhs_results = analyze_hhs_regions()
political_results = analyze_political_control()
political_results.to_csv("political_results.csv", index=False)

death_results = merge_pop(covid_results, hhs_results, political_results)
death_results.to_csv("death_results.csv", index=False)

