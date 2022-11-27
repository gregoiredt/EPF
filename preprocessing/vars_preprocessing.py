DIC_LAGS = {
    # "Prev_J2_Load": [1],  # Prévision du jour même
    "SpotPrice": [1, 7],
    # "ES_SpotPrice": [2, 7],
    # "BE_SpotPrice": [2, 7],
    # "CH_SpotPrice": [2, 7],
    # "IT_SpotPrice": [2, 7],
    # "DE_SpotPrice": [2, 7],
    "GazPrice": [1, 7],
    # "Load": [2, 7],
    "GBP_EUR_SPOT": [1],
    "USD_EUR_SPOT": [1],
    # "Biomass": [2, 7],
    "Fossil_Gas": [2, 7],
    "Fossil_Hard_Coal": [2, 7],
    "Fossil_Oil": [2, 7],
    "Hydro_Pumped_Storage": [2, 7],
    "Hydro_Run-of-river_and_poundage": [2],
    "Hydro_Water_Reservoir": [2, 7],
    "Nuclear": [2, 7],
    # "Solar": [2, 7],
    # "Waste": [2, 7],
    # "Wind_Onshore": [2, 7],
    # "Exchange_FR_BE": [2, 7],
    # "Exchange_FR_CH": [2, 7],
    # "Exchange_FR_ES": [2, 7],
    # "Exchange_FR_GB": [2, 7],
    # "Exchange_FR_IT_North": [2, 7],
    "Exchange_FR_DE": [2, 7],
    'Exchange_FR_TOT': [2, 7]
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

DIC_VAR_CLASSIF = {
    'autoregressive_central_hourly': [
        # "ES_SpotPrice",
        # "BE_SpotPrice",
        # "CH_SpotPrice",
        # "IT_SpotPrice",
        # "UK_SpotPrice",
        # "DE_SpotPrice"
    ],

    'autoregressive_central_daily': [
        # "GazPrice"
    ],

    'exogenous_central_hourly_valid': [
        "Prev_J2_Load",
        "Prev_J1_Solar",
        "Prev_J1_WindOnshore",

    ],

    'exogenous_central_hourly_invalid': [
        # "Load",
        'SpotPrice',
    ],

    "exogenous_additional_hourly_valid": [
        # "Fossil_Hard_Coal_availability",
        # "Fossil_Oil_availability",
        # "Fossil_Gas_availability",
        "Nuclear_availability",
        # "Hydro_Pumped_Storage_availability",
        # "Hydro_Water_Reservoir_availability",
        # "Hydro_Run-of-river_and_poundage_availability",
    ],

    "exogenous_additional_hourly_invalid": [
        # "Biomass",
        "Fossil_Gas",
        "Fossil_Hard_Coal",
        "Fossil_Oil",
        "Hydro_Pumped_Storage",
        "Hydro_Run-of-river_and_poundage",
        "Hydro_Water_Reservoir",
        "Nuclear",
        # "Solar",
        # "Waste",
        # "Wind_Onshore",
        # 'Exchange_FR_BE',
        # 'Exchange_FR_CH',
        'Exchange_FR_DE',
        # 'Exchange_FR_ES',
        # 'Exchange_FR_GB',
        # 'Exchange_FR_IT_North',
        'Exchange_FR_TOT'
    ],

    "exogenous_additional_daily_valid": [
        'M1_Coal',
        'M1_Oil',
        'PublicHoliday_FR',
        # 'PublicHoliday',
        # 'PublicHoliday_DE',
        # 'PublicHoliday_GB',
        # 'PublicHoliday_IT',
        # 'PublicHoliday_BE',
        'SchoolHoliday_FR',
        # 'SchoolHoliday_FR_C',
        # 'SchoolHoliday_FR_A',
        # 'SchoolHoliday_FR_B',
        'weekday',
        'Ponts_FR',
        # 'toy',
        'toy_cos',
        'toy_sin',
        'clock'
    ],

    "exogenous_additional_daily_invalid": [
        'GBP_EUR_SPOT',
        'USD_EUR_SPOT',
        'GazPrice'
    ]

}
