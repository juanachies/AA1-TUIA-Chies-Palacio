import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_group_median = ['Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                     'Evaporation', 'Sunshine', 'Humidity9am', 'Humidity3pm',
                                     'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
        self.temp_group_mean = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']
        self.wind_cols_region_median = ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
        self.wind_dir_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']


    def fit(self, X, y=None):
        df = X.copy()

        self.summary_ = {
            'rows_initial_train': len(df),
            'outliers_replaced_nan': {},
            'dropped_rows': {},
        }

        # outliers
        self.outlier_limits = {}
        for col in ['Humidity9am', 'Humidity3pm', 'Cloud9am', 'Cloud3pm']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_limits[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # Rainfall
        self.rainfall_mean_rd = df.groupby(['Region', 'Date'])['Rainfall'].mean().to_dict()

        # numericas (mediana por Region, Date) 
        self.group_median = {}
        for col in [
            'WindGustSpeed','WindSpeed9am','WindSpeed3pm',
            'Pressure9am','Pressure3pm',
            'Humidity9am','Humidity3pm',
            'Cloud9am','Cloud3pm',
            'Evaporation'
        ]:
            if col in df.columns:
                self.group_median[col] = df.groupby(['Region','Date'])[col].median().to_dict()

        # sunshine (mediana en cascada: Region+Date, Region, Global) 
        if 'Sunshine' in df.columns:
            # Nivel 1: Mediana por (Region, Date)
            self.sunshine_median_rd = df.groupby(['Region','Date'])['Sunshine'].median().to_dict()
            # Nivel 2: Mediana por Region (fallback)
            self.sunshine_median_r = df.groupby('Region')['Sunshine'].median().to_dict()
            # Nivel 3: Mediana global (último fallback)
            self.sunshine_median_global = df['Sunshine'].median()

        #  TEMPERATURAS (media por Region, Date) 
        self.group_mean = {}
        for col in ['MinTemp','MaxTemp','Temp9am','Temp3pm']:
            if col in df.columns:
                self.group_mean[col] = df.groupby(['Region','Date'])[col].mean().to_dict()

        #  DIRECCIÓN DE VIENTO (moda por Region, Date) 
        self.group_mode = {}
        for col in ['WindGustDir','WindDir9am','WindDir3pm']:
            if col in df.columns:
                self.group_mode[col] = df.groupby(['Region','Date'])[col].apply(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                ).to_dict()

        self.summary_['rows_final_train'] = len(df)
        return self


    def transform(self, X):
        df = X.copy()
        rows_start = len(df)

        # ================= OUTLIERS =================
        for col, (low, high) in self.outlier_limits.items():
            if col in df.columns:
                mask = (df[col] < low) | (df[col] > high)
                self.summary_['outliers_replaced_nan'][col] = int(mask.sum())
                df.loc[mask, col] = np.nan

        # ================= RAINFALL =================
        if {'Rainfall','RainToday','Region','Date'}.issubset(df.columns):
            rf_temp = df.apply(
                lambda r: self.rainfall_mean_rd.get((r['Region'], r['Date']), np.nan),
                axis=1
            )

            df.loc[df['RainToday']=='No','Rainfall'] = df.loc[
                df['RainToday']=='No','Rainfall'
            ].fillna(0)

            mask = df['Rainfall'].isna() & ((df['RainToday']=='Yes') | (df['RainToday'].isna()))
            df.loc[mask,'Rainfall'] = rf_temp[mask]

            df.loc[(df['RainToday']=='Yes') & (df['Rainfall']==0),'Rainfall'] = 1

            df['RainToday'] = (df['Rainfall'] > 1).map({True:'Yes',False:'No'})

            self.summary_['dropped_rows']['Rainfall'] = int(df['Rainfall'].isna().sum())
            df = df.dropna(subset=['Rainfall'])

        # ================= NUMÉRICAS (mediana) =================
        for col, stats in self.group_median.items():
            if col in df.columns:
                df.loc[df[col].isna(), col] = df[df[col].isna()].apply(
                    lambda r: stats.get((r['Region'], r['Date']), np.nan), axis=1
                )
                self.summary_['dropped_rows'][col] = int(df[col].isna().sum())
                df = df.dropna(subset=[col])

        # ================= SUNSHINE (mediana en cascada) =================
        if 'Sunshine' in df.columns:
            # Nivel 1: Intentar con (Region, Date)
            df.loc[df['Sunshine'].isna(), 'Sunshine'] = df[df['Sunshine'].isna()].apply(
                lambda r: self.sunshine_median_rd.get((r['Region'], r['Date']), np.nan), axis=1
            )
            
            # Nivel 2: Si aún hay nulos, usar solo Region
            df.loc[df['Sunshine'].isna(), 'Sunshine'] = df[df['Sunshine'].isna()].apply(
                lambda r: self.sunshine_median_r.get(r['Region'], np.nan), axis=1
            )
            
            # Nivel 3: Si AÚN hay nulos, usar mediana global
            df.loc[df['Sunshine'].isna(), 'Sunshine'] = self.sunshine_median_global
            

        # ================= TEMPERATURAS (media) =================
        for col, stats in self.group_mean.items():
            if col in df.columns:
                df.loc[df[col].isna(), col] = df[df[col].isna()].apply(
                    lambda r: stats.get((r['Region'], r['Date']), np.nan), axis=1
                )
                self.summary_['dropped_rows'][col] = int(df[col].isna().sum())
                df = df.dropna(subset=[col])

        # ================= CATEGÓRICAS (moda) =================
        for col, stats in self.group_mode.items():
            if col in df.columns:
                df.loc[df[col].isna(), col] = df[df[col].isna()].apply(
                    lambda r: stats.get((r['Region'], r['Date']), np.nan), axis=1
                )
                self.summary_['dropped_rows'][col] = int(df[col].isna().sum())
                df = df.dropna(subset=[col])

        self.summary_['rows_initial_transform'] = rows_start
        self.summary_['rows_final_transform'] = len(df)

        return df