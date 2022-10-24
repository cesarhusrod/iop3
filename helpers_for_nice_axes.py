import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mplt
import matplotlib.pyplot as plt

from astropy.time import Time
    
def set_ax_jyear(ax):
        # get limits
        xlims = ax.get_xlim()
        
        # get ticks and format them
        xticks = ax.get_xticks()
        
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{jyear:.5f}" for jyear in Time(xticks, format='jyear').jyear]) 
        ax.set_xlabel('Year')

        # restore limits
        ax.set_xlim(xlims)
        
def set_ax_dates(ax, ax_date, nmax_hours=72, nmax_days=90, nmax_months=12, labels=True):  
        # get limits
        xlims = ax.get_xlim()
        
        # get ticks and format them
        import datetime

        xlims_0_dt = Time(xlims[0], format='jyear').datetime
        xlims_1_dt = Time(xlims[1], format='jyear').datetime

        
        if np.ptp(ax.get_xlim()) < nmax_hours/24 * 1/365:
            step = (np.ptp(ax.get_xlim()) * 365 * 24 / 10)
            xticks_date = Time(np.arange(xlims_0_dt-datetime.timedelta(hours=step), 
                                         xlims_1_dt+datetime.timedelta(hours=step),
                                         datetime.timedelta(hours=step)
                                        ).astype(datetime.datetime)).jyear
            date_fmt = '%Y-%m-%d \n %H:%M'
        elif np.ptp(ax.get_xlim()) < nmax_days * 1/365:
            step = np.ptp(ax.get_xlim()) * 365 // 10 + 1
            xticks_date = Time(np.arange(datetime.date(xlims_0_dt.year, xlims_0_dt.month, xlims_0_dt.day) - datetime.timedelta(days=step), 
                                         datetime.date(xlims_1_dt.year, xlims_1_dt.month, xlims_1_dt.day) + datetime.timedelta(days=step),
                                         datetime.timedelta(days=step)
                                        ).astype(datetime.datetime)).jyear
            date_fmt = '%Y-%m-%d'
        elif np.ptp(ax.get_xlim()) < nmax_months * 1/12:
            step = np.ptp(ax.get_xlim()) * 12 // nmax_months + 1
            xticks_date = Time(pd.date_range(start=datetime.date(xlims_0_dt.year, xlims_0_dt.month, 1),
                                             end=datetime.date(xlims_1_dt.year, xlims_1_dt.month, 1),
                                             freq=f"{step:.0f}M")).jyear
            date_fmt = '%b %Y'
        else:
            step = np.ptp(ax.get_xlim()) // 12 + 1
            xticks_date = Time(pd.date_range(start=datetime.date(xlims_0_dt.year, 1, 1),
                                             end=datetime.date(xlims_1_dt.year+1, 1, 1),
                                            freq=f"{step:.0f}Y")).jyear
            date_fmt = '%Y'
            
        ax_date.set_xticks(xticks_date)
        if labels:
            xticks_date_labels = Time(xticks_date, format='jyear').strftime(date_fmt)
        else:
            xticks_date_labels = ['' for xtick in xticks_date]

        ax_date.set_xticklabels(xticks_date_labels)
        
        # restore limits
        ax_date.set_xlim(xlims)
        
        return ax_date
    
    
    
    
    
def set_ax_mjd(ax):
        # get limits
        xlims = ax.get_xlim()
        
        # get ticks and format them
        xticks = ax.get_xticks()
        
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{mjd:.3f}" for mjd in Time(xticks, format='mjd').mjd]) 
        ax.set_xlabel('MJD')

        # restore limits
        ax.set_xlim(xlims)
        
def set_ax_dates_mjd(ax, ax_date, nmax_hours=72, nmax_days=90, nmax_months=12, labels=True):  
        # get limits
        xlims = ax.get_xlim()
        
        # get ticks and format them
        import datetime

        xlims_0_dt = Time(xlims[0], format='mjd').datetime
        xlims_1_dt = Time(xlims[1], format='mjd').datetime

        
        if np.ptp(ax.get_xlim())/365 < nmax_hours/24 * 1/365:
            step = (np.ptp(ax.get_xlim())/365 * 365 * 24 / 9)
            xticks_date = Time(np.arange(xlims_0_dt-datetime.timedelta(hours=step), 
                                         xlims_1_dt+datetime.timedelta(hours=step),
                                         datetime.timedelta(hours=step)
                                        ).astype(datetime.datetime)).mjd
            date_fmt = '%Y-%m-%d \n %H:%M'
        elif np.ptp(ax.get_xlim())/365 < nmax_days * 1/365:
            step = np.ptp(ax.get_xlim())/365 * 365 // 9 + 1
            xticks_date = Time(np.arange(datetime.date(xlims_0_dt.year, xlims_0_dt.month, xlims_0_dt.day) - datetime.timedelta(days=step), 
                                         datetime.date(xlims_1_dt.year, xlims_1_dt.month, xlims_1_dt.day) + datetime.timedelta(days=step),
                                         datetime.timedelta(days=step)
                                        ).astype(datetime.datetime)).mjd
            date_fmt = '%Y-%m-%d'
        elif np.ptp(ax.get_xlim())/365 < nmax_months * 1/12:
            step = np.ptp(ax.get_xlim())/365 * 12 // nmax_months + 1
            xticks_date = Time(pd.date_range(start=datetime.date(xlims_0_dt.year, xlims_0_dt.month, 1),
                                             end=datetime.date(xlims_1_dt.year, xlims_1_dt.month, 1),
                                             freq=f"{step:.0f}M")).mjd
            date_fmt = '%b %Y'
        else:
            step = np.ptp(ax.get_xlim())/365 // 12 + 1
            xticks_date = Time(pd.date_range(start=datetime.date(xlims_0_dt.year, 1, 1),
                                             end=datetime.date(xlims_1_dt.year+1, 1, 1),
                                            freq=f"{step:.0f}Y")).mjd
            date_fmt = '%Y'
         
        
        ax_date.set_xticks(xticks_date)
        if labels:
            xticks_date_labels = Time(xticks_date, format='mjd').strftime(date_fmt)
        else:
            xticks_date_labels = ['' for xtick in xticks_date]

        ax_date.set_xticklabels(xticks_date_labels)

        # restore limits
        ax_date.set_xlim(xlims)
        
        return ax_date