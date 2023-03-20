def TimeDistributionInsights(time_dim,user_type,cat_dim):
    
    if time_dim == 'Hour' and user_type == 'Total Users' and cat_dim == 'None':
        insight = 'The moments in which bikes-sharing service is used the most are around 8:00 and 17:00-18:00. Between the two peaks, the usage of bikes lowers and has a constant behaviour. Also, we can see that people tend to use the bike-sharing service at a higher rate in the afternoon rather than in the morrning. Probably this is related to peak hours where people are commuting to work or school.'
        
    elif time_dim == 'Hour' and user_type == 'Total Users' and cat_dim == 'Weekday':
        insight = 'In the weekdays, the behaviour of bike-sharing service is the same for every day, where the peak hours are at 8:00 and between 17:00 and 18:00, with a lower and constant usage between these two peaks. Nonetheless, in the weekend, the behaviour changes and the bike-sharing service starts increasing its usage between 8:00 and 9:00, reaches its highest rate between 13:00 and 15:00 in the afternoon and then starts to slowly decrease. This could be associated to the activities that people do during the weekend differing to those that people to during the week (work/study).'

    elif time_dim == 'Month' and user_type == 'Total Users' and cat_dim == 'None':
        insight = 'The bike-sharing service has a slow increase in its usage as the year passes, with its lowest rate in january. In June it gets to the higer rate of usage and mantains there (with some variations) for the next 4 months. After this, it slowly starts to decay. This could be associated to the seasons and their respective climate.'
        
    elif time_dim == 'Total' and user_type == 'Total Users' and cat_dim == 'Weather Condition':
        insight = 'As expected, we can conclude that the bike-sharing service is mostly used when the weather condition is clear, has mist or is cloudy. In this sense, the average usage of the bike-sharing service is quite higher when the weather conditions are sunny versus when there is mist/cloudy. In the other hand, also as expected, when the weather conditions include rain or snow the bike-sharing service lowers with the average tending to zero as the weather conditions get worse.'
    
    elif time_dim == 'Total' and user_type == 'Total Users' and cat_dim == 'Is Working Day':
        insight = 'We can see that people prefer to use the bike-sharing service during the weekdays rather than on the weekend or on a holiday.'
    else:
        insight = None
        
    return insight