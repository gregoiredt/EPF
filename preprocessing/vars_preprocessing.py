DIC_LAGS = {
    "Prev_J2_Load": [1],  # Prévision du jour même
    "SpotPrice": [2, 7],
    "ES_SpotPrice": [2, 7],
    "BE_SpotPrice": [2, 7],
    "CH_SpotPrice": [2, 7],
    "IT_SpotPrice": [2, 7],
    "DE_SpotPrice": [2, 7],
    "GazPrice": [1, 7],
    "Load": [2, 7],
    "GBP_EUR_SPOT": [1],
    "USD_EUR_SPOT": [1],
    "Biomass": [2, 7],
    "Fossil_Gas": [2, 7],
    "Fossil_Hard_Coal": [2, 7],
    "Fossil_Oil": [2, 7],
    "Hydro_Pumped_Storage": [2, 7],
    "Hydro_Run-of-river_and_poundage": [2, 7],
    "Hydro_Water_Reservoir": [2, 7],
    "Nuclear": [2, 7],
    "Solar": [2, 7],
    "Waste": [2, 7],
    "Wind_Onshore": [2, 7],
    "Exchange_FR_BE": [2, 7],
    "Exchange_FR_CH": [2, 7],
    "Exchange_FR_ES": [2, 7],
    "Exchange_FR_GB": [2, 7],
    "Exchange_FR_IT_North": [2, 7],
    "Exchange_FR_DE": [2, 7]
}

PIB_PERC = {
    "FR": 2.78 / 12.23,
    "DE": 3.96 / 12.23,
    "GB": 2.86 / 12.23,
    "IT": 2.09 / 12.23,
    "BE": 0.54 / 12.23
}

PERC_POP_ZONE = {
    'A': 22.715 / 60.387,
    'B': 23.246 / 60.387,
    'C': 14.426 / 60.387
}

LIST_FOREIGN_COUNTRIES = ['BE', 'CH', 'DE', 'GB', 'IT_North', 'ES']

DIC_COL_INDISPO = {
    "biomass": "Biomass_availability",
    "fossil_coal": "Fossil_Hard_Coal_availability",
    "fossil_oil": "Fossil_Oil_availability",
    "fossil_gas": "Fossil_Gas_availability",
    "hydro_pumped_storage": "Hydro_Pumped_Storage_availability",
    "hydro_reservoir": 'Hydro_Water_Reservoir_availability',
    "hydro_run_of_river": "Hydro_Run-of-river_and_poundage_availability",
    "marine": "Marine_availability",
    "nuclear": 'Nuclear_availability',
    "other": 'Other_availability',
}
