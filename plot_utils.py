from utils import *

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set(font_scale=2, style='white')

# Source: https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
def clean_plot(ax):

    plt.tight_layout()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def dates_xaxis(ax, frequency):

    if frequency == 'day':
        locator = mdates.DayLocator()
        format = mdates.DateFormatter('%y-%m-%d')  

    elif frequency == 'week':
        locator = mdates.WeekdayLocator()  
        format = mdates.DateFormatter('%y-%m-%d')

    elif frequency == 'month':
        locator = mdates.MonthLocator()  
        format = mdates.DateFormatter('%y-%m')

    elif frequency == 'year':
        locator = mdates.YearLocator()
        format = mdates.DateFormatter('%Y')

    else:
        raise ValueError('Invalid frequency for date axis.')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(format)