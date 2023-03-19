def TimeDistributionInsights(time_dim,user_type,cat_dim):
    
    if time_dim == 'Hour' and user_type == 'Total Users' and cat_dim == 'None':
        insight = 'This is an insight for hour, total users and none cat dim'
        
    elif time_dim == 'Hour' and user_type == 'Total Users' and cat_dim == 'Weather Condition':
        insight = 'This is an insight for hour, total users and weather condition'

    else:
        insight = None
        
    return insight