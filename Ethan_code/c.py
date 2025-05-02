import pandas as pd
from collections import defaultdict
import csv

def clean_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Step 1: Drop duplicates
    df = df.drop_duplicates()

    # Step 2: Handle missing values
    # Drop columns where more than 50% values are missing
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Fill remaining missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill remaining missing object (string) values with empty string
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("")

    # Step 3: Convert data types (optional)
    # Example: convert object columns that look like numbers to numeric
    for col in object_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Skip columns that can't be converted

    return df

def analyze_covid_deaths(df):
    df = df[df["jurisdiction_residence"] != "United States"]
    df = df[df["group"] == "Race and Age"]
    std_df = pd.DataFrame()
    time_col = pd.to_datetime(
    df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )

    min_time = time_col.min()
    # Save months since min_time
    std_df['time'] = (time_col.dt.year - min_time.year) * 12 + (time_col.dt.month - min_time.month)
    std_df['year'] = df['year']
    age_map = {
        '0-4 years': 0, '5-11 years': 1, '12-17 years': 2, '18-29 years': 3,
        '30-39 years': 4, '40-49 years': 5, '50-64 years': 6, 
        '65-74 years': 7, '75 years and over': 8
    }

    std_df['age'] = df['subgroup2'].map(age_map)
    std_df['race'] = df['subgroup1']
    std_df['region'] = df['jurisdiction_residence']
    std_df['death_rate'] = df['crude_COVID_rate']
    std_df = std_df.sort_values(by='time')
    return std_df

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

def analyze_political_control():
    """Analyze political control percentages by region and year"""
    # Load political control data
    political_df = pd.read_csv('attached_assets/state_political_control_2020_2025.csv')

    # Dictionary mapping states to HHS regions
    state_to_region = {}
    for region, states in hhs_regions.items():
        for state in states:
            state_to_region[state] = region

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

        results.append(f"\nPolitical Control Analysis for {year}:")
        for region in sorted(region_totals.keys(), key=int):
            total = region_totals[region]
            stats = region_stats[region]
            results.append(f"\nRegion {region}:")
            results.append(f"Legislature: Republican {stats['rep_leg']/total*100:.1f}%, Democrat {stats['dem_leg']/total*100:.1f}%, Mixed {stats['mix_leg']/total*100:.1f}%")
            results.append(f"Governor: Republican {stats['rep_gov']/total*100:.1f}%, Democrat {stats['dem_gov']/total*100:.1f}%")
            results.append(f"State Control: Republican {stats['rep_state']/total*100:.1f}%, Democrat {stats['dem_state']/total*100:.1f}%, Mixed {stats['mix_state']/total*100:.1f}%")

    return results


#Main function that runs when the script is executed directly
covid_results = analyze_covid_deaths()
political_results = analyze_political_control()


# Load the COVID data
covid_df = pd.read_csv("attached_assets/Provisional_COVID-19_death_counts_and_rates_by_month__jurisdiction_of_residence__and_demographic_characteristics_20250415.csv")

# Load population data
pop_df = pd.read_csv("attached_assets/NST-EST2024-ALLDATA.csv")

# Load political control data
political_df = pd.read_csv("attached_assets/state_political_control_2020_2025.csv")

# Initialize population data dictionary
pop_data = {}
for year in range(2020, 2026):
    year_str = str(year)
    col = f'POPESTIMATE{year}' if year != 2025 else 'ESTIMATESBASE2020'
    pop_data[year_str] = pop_df[pop_df['SUMLEV'] == '040'].set_index('NAME')[col].to_dict()

# Create rows for the combined dataset
combined_rows = []
for year in range(2020, 2026):
    year_str = str(year)
    for region in hhs_regions:
        deaths = 0
        pop = 0
        stats = defaultdict(int)
        total = 0

        for state in hhs_regions[region]:
            # Get COVID deaths for this state and year
            state_deaths = 0
            death_data = covid_df[
                (covid_df['group'] == 'Sex') & 
                (covid_df['year'] == str(year)) &
                (covid_df['jurisdiction_residence'] == state)
            ]['COVID_deaths']
            
            if not death_data.empty:
                try:
                    state_deaths = float(death_data.fillna(0).sum())
                except (ValueError, TypeError):
                    state_deaths = 0.0
            
            # Get population data as float
            try:
                state_pop = float(pop_data[year_str].get(state, 0))
            except (ValueError, TypeError):
                state_pop = 0.0
                
            deaths += state_deaths
            pop += state_pop

            # Get political control
            pol_data = political_df[
                (political_df['Year'] == year) & 
                (political_df['State'] == state)
            ]

            if not pol_data.empty:
                total += 1
                row = pol_data.iloc[0]

                # Count controls
                if row['Legislature_Control'] == 2:
                    stats['rep_leg'] += 1
                elif row['Legislature_Control'] == 1:
                    stats['dem_leg'] += 1
                else:
                    stats['mix_leg'] += 1

                if row['Governor_Control'] == 2:
                    stats['rep_gov'] += 1
                elif row['Governor_Control'] == 1:
                    stats['dem_gov'] += 1

                if row['State_Control'] == 2:
                    stats['rep_state'] += 1
                elif row['State_Control'] == 1:
                    stats['dem_state'] += 1
                else:
                    stats['mix_state'] += 1

            deaths += state_deaths
            pop += state_pop

        if total > 0:
            # Get index for the current region in the lists (0-9 for regions 1-10)
            list_index = (int(region) - 1) + ((year - 2020) * 10)
            
            if list_index < len(deaths_list):
                row = {
                    "Year": year,
                    "Region": int(region),
                    "Deaths": deaths_list[list_index],
                    "Population": populations_list[list_index],
                    "%Death_Rate": death_rates_list[list_index],
                    "%Rep_Leg": (stats['rep_leg'] / total * 100),
                "%Dem_Leg": (stats['dem_leg'] / total * 100),
                "%Mix_Leg": (stats['mix_leg'] / total * 100),
                "%Rep_Gov": (stats['rep_gov'] / total * 100),
                "%Dem_Gov": (stats['dem_gov'] / total * 100),
                "%Rep_State": (stats['rep_state'] / total * 100),
                "%Dem_State": (stats['dem_state'] / total * 100),
                "%Mix_State": (stats['mix_state'] / total * 100)
            }
            combined_rows.append(row)

# Create the combined DataFrame
combined_df = pd.DataFrame(combined_rows)

# Save to CSV
output_path = 'attached_assets/combined_dataset.csv'
combined_df.to_csv(output_path, index=False)
print("\nCombined Dataset saved to:", output_path)
print("\nCombined Dataset Preview:")
print(combined_df)