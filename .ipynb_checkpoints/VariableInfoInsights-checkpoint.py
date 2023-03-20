def VariableInfoInsights(variable):
        
    if variable == 'Season':
        insight = 'Its a categorical variable with 4 different categories (seasons). The one that has the higher bike-sharing service is season 3 (summer), followed by season 2 (Spring), then season 4 (fall), and finally season 1 (winter).'

    elif variable == 'Year':
        insight = 'Its a categorical variable with 2 different categories. It shows that the distribution is equally divided between the two years.'
    
    elif variable == 'Month':
        insight = 'Its a categorical variable with 12 different categories, one for each month.'
        
        
    elif variable == 'Is_Working_Day':
        insight = 'Its a categorical variable with 2 different categories. The bike-sharing service doubles the rate of usage in the weekday (category 1) versus not-weekday or holiday (category 0).'
        
    elif variable == 'Weather_Condition':
        insight = 'Its a categorical variable with 3 different categories. As weather conditions are better, the bike-sharing service rate is also better.'
        
    elif variable == 'Temperature':
        insight = 'At extreme high or extreme low outside temperatures, the bike-sharing service decreases.'
        
    elif variable == 'Temperature_Feel':
        insight = 'At extreme high or extreme low feeling temperatures, the bike-sharing service decreases.'
        
    elif variable == 'Wind_Speed':
        insight = 'The distribution for Wind Speed is skewed to the right which means that as wind speed is lower, the bike-sharing service increases.'
        
    elif variable == 'Casual_Users':
        insight = 'We can see that the data is right skewed which tells us that the bike-service sharing rate is low.'
        
    elif variable == 'Registered_Users':
        insight = 'We can see that the data is right skewed which tells us that the bike-service sharing rate is low.'
        
    elif variable == 'Total_Users':
        insight = 'We can see that the data is right skewed which tells us that the bike-service sharing rate is low.'
    
    else:
        insight = None
        
    return insight