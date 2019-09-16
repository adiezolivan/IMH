from datetime import datetime
from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

independent_labels = ['Caudal_naphta','Caudal_vapor','Temp_Coil_4_htc1','Temp_Coil_3_htc1','Temp_Coil_2_htc1','Temp_Coil_1_htc1','Temp_Coil_5_htc1',
    'Temp_Coil_6_htc1',
    'Temp_Coil_7_htc1',
    'Temp_Coil_8_htc1',
    'Temp_Coil_1_htc2',
    'Temp_Coil_2_htc2',
    'Temp_Coil_3_htc2',
    'Temp_Coil_4_htc2',
    'Temp_Coil_8_htc2',
    'Temp_Coil_7_htc2',
    'Temp_Coil_6_htc2',
    'Temp_Coil_5_htc2',
    'Temp_Sal_Humos_1',
    'Temp_Sal_Humos_2',
    'Temp_Sal_Humos_3',
    'Temp_RG_ENT_B',
    'Temp_RG_ENT_A',
    'Pres_RG_ENT_A',
    'Pres_RG_ENT_B',
    'Pres_Coil_1',
    'Pres_Coil_2',
    'Pres_Coil_3',
    'Pres_Coil_4',
    'Pres_Coil_8',
    'Pres_Coil_7',
    'Pres_Coil_6',
    'Pres_Coil_5',
    'Densidad_Nafta']

translation = OrderedDict([('ED_AI_0364','Caudal_naphta'),
    ('ED_AI_0395','Caudal_vapor'),
    ('ED_AI_0339','Temp_Coil_4_htc1'),
    ('ED_AI_0338','Temp_Coil_3_htc1'),
    ('ED_AI_0337','Temp_Coil_2_htc1'),
    ('ED_AI_0336','Temp_Coil_1_htc1'),
    ('ED_AI_0436','Temp_Coil_5_htc1'),
    ('ED_AI_0437','Temp_Coil_6_htc1'),
    ('ED_AI_0438','Temp_Coil_7_htc1'),
    ('ED_AI_0439','Temp_Coil_8_htc1'),
    ('ED_AI_0316','Temp_Coil_1_htc2'),
    ('ED_AI_0317','Temp_Coil_2_htc2'),
    ('ED_AI_0318','Temp_Coil_3_htc2'),
    ('ED_AI_0319','Temp_Coil_4_htc2'),
    ('ED_AI_0419','Temp_Coil_8_htc2'),
    ('ED_AI_0418','Temp_Coil_7_htc2'),
    ('ED_AI_0417','Temp_Coil_6_htc2'),
    ('ED_AI_0416','Temp_Coil_5_htc2'),
    ('ED_AI_0387','Temp_Sal_Humos_1'),
    ('ED_AI_0397','Temp_Sal_Humos_2'),
    ('ED_AI_0487','Temp_Sal_Humos_3'),
    ('ED_AI_0410','Temp_RG_ENT_B'),
    ('ED_AI_0310','Temp_RG_ENT_A'),
    ('ED_AI_0343','Pres_RG_ENT_A'),
    ('ED_AI_0443','Pres_RG_ENT_B'),
    ('ED_AI_0366','Pres_Coil_1'),
    ('ED_AI_0367','Pres_Coil_2'),
    ('ED_AI_0368','Pres_Coil_3'),
    ('ED_AI_0369','Pres_Coil_4'),
    ('ED_AI_0469','Pres_Coil_8'),
    ('ED_AI_0468','Pres_Coil_7'),
    ('ED_AI_0467','Pres_Coil_6'),
    ('ED_AI_0466','Pres_Coil_5'),
    ('EI_AI_0110','Densidad_Nafta'),
    ('EH_AC_3034','H2_A'),
    ('EH_AC_3035','CH4_A'),
    ('EH_AC_3037','C2H4_A'),
    ('EH_AC_3040','C3H6_A'),
    ('EH_AC_3053','H2_B'),
    ('EH_AC_3054','CH4_B'),
    ('EH_AC_3056','C2H4_B'),
    ('EH_AC_3059','C3H6_B')
])
all_labels = ['Caudal_naphta','Caudal_vapor','Temp_Coil_4_htc1','Temp_Coil_3_htc1','Temp_Coil_2_htc1','Temp_Coil_1_htc1','Temp_Coil_5_htc1',
    'Temp_Coil_6_htc1',
    'Temp_Coil_7_htc1',
    'Temp_Coil_8_htc1',
    'Temp_Coil_1_htc2',
    'Temp_Coil_2_htc2',
    'Temp_Coil_3_htc2',
    'Temp_Coil_4_htc2',
    'Temp_Coil_8_htc2',
    'Temp_Coil_7_htc2',
    'Temp_Coil_6_htc2',
    'Temp_Coil_5_htc2',
    'Temp_Sal_Humos_1',
    'Temp_Sal_Humos_2',
    'Temp_Sal_Humos_3',
    'Temp_RG_ENT_B',
    'Temp_RG_ENT_A',
    'Pres_RG_ENT_A',
    'Pres_RG_ENT_B',
    'Pres_Coil_1',
    'Pres_Coil_2',
    'Pres_Coil_3',
    'Pres_Coil_4',
    'Pres_Coil_8',
    'Pres_Coil_7',
    'Pres_Coil_6',
    'Pres_Coil_5',
    'H2_B', 'CH4_B', 'C2H4_B', 'C3H6_B']


def dateparse (time):
    try:
        return datetime.strptime(time,"%Y/%d/%m %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(time,"%Y/%d/%m")
        except Exception:
            try:
                return datetime.strptime(time,"%d/%m/%Y")
            except Exception:
                try:
                    return datetime.strptime(time,"%d/%m/%Y %H:%M:%S")
                except Exception:
                    return datetime.strptime(time, "%Y/%m/%d")


def load_long_data():
    file_name = 'DATOS-SOFT-SENSOR_OCT-DIC_2016_v3.csv'
    df = pd.read_csv(file_name, sep=',', header=0, index_col=0, parse_dates=True, date_parser=dateparse)
    df = df.rename(columns=translation)

    return df


def load_short_data():
    file_name = 'DATOS-SOFT-SENSOR_OCT-DIC_2016_v3.csv'
    df = pd.read_csv(file_name, sep=',', header=0, index_col=0, parse_dates=True, date_parser=dateparse)
    df = df.rename(columns=translation)

    return df


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = abs(points - median)
   # diff = np.sqrt(diff)
    med_abs_deviation = 1.4826 * np.median(diff)

    #modified_z_score = abs(points - median) / med_abs_deviation
    modified_z_score = abs(points - median) / np.std(points)


    return modified_z_score > thresh

def smooth_data(data):
    smoothing = pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        for output in data.columns:
            savgol_smooth = savgol_filter(data[output].values, 31, 5)
            smoothing[output] = savgol_smooth
    else:
        savgol_smooth = savgol_filter(data, 31, 5)
        smoothing['output'] = savgol_smooth

    return smoothing.set_index(data.index)