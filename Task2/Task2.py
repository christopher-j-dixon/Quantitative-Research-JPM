import pandas as pd

def final_fixed_transfer_contract_value(injection_dates, withdrawal_dates, gas_prices, 
                                        transfer_rate, maximum_volume, storage_cost_per_month, 
                                        fixed_transfer_cost=10000):
    """
    Calculates the value of the contract based on the provided parameters.
    
    Parameters:
    - injection_dates (list): List of dates (days) when gas is injected into storage.
    - withdrawal_dates (list): List of dates (days) when gas is withdrawn from storage.
    - gas_prices (DataFrame): Dataframe of gas prices on the given date ($).
    - transfer_rate (float): Rate of gas to that can be injected/withdrawn per day (MMBtu/day).
    - maximum_volume (float): Maximum storage capacity (MMBtu).
    - storage_cost_per_month (float): Cost to store gas per month ($/month).
    - fixed_transfer_cost (float): Fixed cost per transfer event ($).

    Returns:
    (float): Returns the value of the contract ($).
    """
    # Convert all dates to datetime format
    injection_dates = pd.to_datetime(injection_dates)
    withdrawal_dates = pd.to_datetime(withdrawal_dates)
    gas_prices['Dates'] = pd.to_datetime(gas_prices['Dates'])

    # Sort the dates
    injection_dates = sorted(injection_dates)
    withdrawal_dates = sorted(withdrawal_dates)
    
    # Data validation
    if not all(date in gas_prices['Dates'].values for date in injection_dates + withdrawal_dates):
        raise ValueError("Some injection or withdrawal dates do not have corresponding gas prices.")
    
    # Ensure storage capacity is not exceeded
    current_volume = 0
    for date in sorted(set(injection_dates + withdrawal_dates)):
        if date in injection_dates:
            current_volume += transfer_rate
        if date in withdrawal_dates:
            current_volume -= transfer_rate
        if current_volume > maximum_volume:
            raise ValueError('The total volume injected exceeds the storage capacity.')
    
    # Calculate the gas sales revenue
    sales_revenue = 0
    withdrawal_prices = gas_prices[gas_prices['Dates'].isin(withdrawal_dates)]
    withdrawal_prices = withdrawal_prices.groupby('Dates')['Prices'].first().reset_index()
    for _, row in withdrawal_prices.iterrows():
        sales_revenue += row['Prices'] * transfer_rate

    # Calculate the gas purchase cost
    purchase_cost = 0
    injection_prices = gas_prices[gas_prices['Dates'].isin(injection_dates)]
    injection_prices = injection_prices.groupby('Dates')['Prices'].first().reset_index()
    for _, row in injection_prices.iterrows():
        purchase_cost += row['Prices'] * transfer_rate

    # Calculate the cost to withdraw and inject the gas (Fixed cost per transfer event)
    transfer_dates = set(injection_dates + withdrawal_dates)
    transfer_cost = fixed_transfer_cost * len(transfer_dates)

    # Calculate total storage months
    storage_cost = 0
    for date_i in injection_dates:
        relevant_withdrawal_dates = [date_w for date_w in withdrawal_dates if date_w > date_i]
        if relevant_withdrawal_dates:
            date_w = relevant_withdrawal_dates[0]
            months_difference = (date_w.year - date_i.year) * 12 + (date_w.month - date_i.month)
            if date_w.day < date_i.day:
                months_difference -= 1
            storage_cost += months_difference * storage_cost_per_month

    # Total costs from purchase, transfer, and storage
    total_cost = purchase_cost + storage_cost + transfer_cost

    # Calculate the contract's net value
    contract_value_result = sales_revenue - total_cost

    return contract_value_result
