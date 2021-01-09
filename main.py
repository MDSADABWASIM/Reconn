import streamlit as st
import helper
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    image = Image.open('data/Reconn.png')
    st.image(image, use_column_width=False)

    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file)
        return df

    uploaded_file = st.file_uploader('Upload file to begin', type=("csv"))

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        target_column = st.selectbox('Select Target Column', list(df.columns), key='target_column')
        
        st.sidebar.title('Know your dataset')  

        if st.sidebar.checkbox("Preview Dataset"):
            st.subheader('Dataset preview')
            if st.button("Head"):
                st.write(df.head(10))
            elif st.button("Tail"):
                st.write(df.tail(10))
            else:
                number = st.slider("Select No of Rows to show", 10, df.shape[0])
                st.write(df.head(number))
        
        if st.sidebar.checkbox("Show Column Names"):
            st.subheader('Column names')
            st.write(df.columns)

        if st.sidebar.checkbox("Show Dimensions"):
            st.write(df.shape)

        if st.sidebar.checkbox('Describe', value=False):
                st.markdown('## Data Description')
                st.write(df.describe())
                st.markdown('### Columns that are potential binary features')
                bin_cols = []
                for col in df.columns:
                    if len(df[col].value_counts()) == 2:
                        bin_cols.append(col)    
                st.write(bin_cols)
                st.markdown('### Columns Types')  
                st.write(df.dtypes)

        if st.sidebar.checkbox('Missing Data', value=False):
            st.markdown('## Missing Data')
            total = df.isnull().sum().sort_values(ascending=False)
            percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
            st.write(missing_data)
            try:
                sns.heatmap(df.isnull())
                st.pyplot()
            except:
                st.warning('Error when showing plots')
            
        if st.sidebar.checkbox('Value Counts', value=False):
            st.markdown('## Value Counts')
            col = st.selectbox('Select Column', list(df.columns), key='val_col')
            st.write(df[col].value_counts())

        if st.sidebar.checkbox('Unique elements', value=False):
            st.markdown('## Unique elements')
            if st.checkbox('Show all unique elements', value=False):
                st.write(df.nunique())
            col = st.selectbox('Show columnwise unique elements',list(df.columns),key='unique_col')
            st.write(df[col].unique())

        if st.sidebar.checkbox('Show Distribution', False):
            st.subheader(f'Distribution of {target_column}')
            try:
                sns.distplot(df[target_column])
                st.write("Skewness: %.3f" % df[target_column].skew())
                st.write("Kurtosis: %.3f" % df[target_column].kurt())
                st.pyplot()
            except:
                st.error('Invalid Column')
        
        st.sidebar.title('Explore the Dataset')
        
        if target_column is not None:
            if st.sidebar.checkbox('Scatter Plot', value=False):
                scatter_cols = st.sidebar.multiselect('Select Column', list(df.columns), key='scatter_cols')
                st.markdown('## Scatter Plots')
                for col in scatter_cols:
                    try:
                        data = pd.concat([df[target_column], df[col]], axis=1)
                        data.plot.scatter(x=col, y=target_column, ylim=(0,800000))
                        st.pyplot()
                    except:
                        st.error('Invalid column')

            if st.sidebar.checkbox('Box Plot', value=False):
                box_cols = st.sidebar.multiselect('Select Column', list(df.columns), key='box_cols')
                st.markdown('## Box Plots')
                for col in box_cols:
                    try:
                        data = pd.concat([df[target_column], df[col]], axis=1)
                        f, ax = plt.subplots(figsize=(8, 6))
                        fig = sns.boxplot(x=col, y=target_column, data=data)
                        fig.axis(ymin=np.min(df[target_column]), ymax=np.max(df[target_column]))
                        st.pyplot()
                    except:
                        st.error('Invalid column')
            
            if st.sidebar.checkbox('Pair Plot', value=False):
                pair_cols = st.sidebar.multiselect('Select Column', list(df.columns), key='pair_plot')
                plot_size = st.sidebar.number_input('Select Plot size', 1.0, 5.0, step=0.5, key='plot_size', value=2.5)
                st.markdown('## Pair Plots')
                cols = [target_column]
                for col in pair_cols:
                    cols.append(col)
                try:
                    sns.set()
                    sns.pairplot(df[cols], height = plot_size)
                    st.pyplot()
                except:
                    st.error('Invalid column')

            if st.sidebar.checkbox('Correlation matrix', value=False):
                st.markdown('## Correlation matrix (heatmap style)')
                corrmat = df.corr()
                f, ax = plt.subplots(figsize=(12, 9))
                sns.heatmap(corrmat, vmax=.8, square=True)
                st.pyplot()

                if st.checkbox('With Target Column', value=False):
                    k = st.number_input('# of Cols for heatmap', 3, len(df.columns), step=1, key='k') #number of variables for heatmap
                    cols = corrmat.nlargest(k, target_column)[target_column].index
                    cm = np.corrcoef(df[cols].values.T)
                    sns.set(font_scale=1.25)
                    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
                    st.pyplot()

        st.sidebar.title('Data processing')
       
        if st.sidebar.checkbox('Treat missing values'):
            st.subheader('Treat missing values')
            # Select a column to treat missing values
            col_option = st.selectbox("Select Column to treat missing values", df.columns) 

            # Specify options to treat missing values
            missing_values_clear = st.selectbox("Select Missing values treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

            if missing_values_clear == "Replace with Mean":
                replaced_value = df[col_option].mean()
                st.write("Mean value of column is :", replaced_value)
            elif missing_values_clear == "Replace with Median":
                replaced_value = df[col_option].median()
                st.write("Median value of column is :", replaced_value)
            elif missing_values_clear == "Replace with Mode":
                replaced_value = df[col_option].mode()
                st.write("Mode value of column is :", replaced_value)
            
            Replace = st.selectbox("Replace values of column?", ("No","Yes"))
            if Replace == "Yes":
                df[col_option] = df[col_option].fillna(replaced_value)
                st.write("Null values replaced")
            elif Replace == "No":
                st.write("No changes made")

        if st.sidebar.checkbox('Encode categorical column'):
            st.subheader("Encode categorical column")
            # Select a column to do encoding
            col_selected = st.selectbox("Select Column to treat categorical values", df.columns) 

            # Specify options to do encoding
            encoder_type = st.selectbox("Select Encoding method", ("Label Encoder",""))

            if encoder_type == "Label Encoder":
                encoded_value = helper.labelEncoder.fit_transform(df[col_selected])
                st.write("Label Encoded value of column is :", encoded_value)
            # elif encoder_type == "Ordinal Encoder":
            #     encoded_value = helper.ordinalEncoder.fit_transform(df[col_selected])
            #     st.write("Ordinal Encoded value of column is :", encoded_value)
            
            Replace = st.selectbox("Replace values of column?", ("No","Yes"),key='encoder')
            if Replace == "Yes":
                df[col_selected] = encoded_value
                st.write("Added encoded column in dataframe")
                st.write(df.head())
            elif Replace == "No":
                st.write('No values replaced yet')

        if st.sidebar.checkbox('Scale column'):
            st.subheader("Scaling column")
            col_scaled = st.selectbox("Select Column for feature scaling", df.columns) 

            scaler_type = st.selectbox("Select Scaling method", ("Standard Scaler","Min Max Scaler"))

            if scaler_type == "Standard Scaler":
                scaled_value = helper.standartScaler.fit_transform(df[col_scaled].values.reshape(-1,1))
                st.write("Standard scaled value of column is :", scaled_value)
            elif scaler_type == "Min Max Scaler":
                scaled_value = helper.minMaxScaler.fit_transform(df[col_scaled].values.reshape(-1,1))
                st.write("Min-Max scaled value of column is :", scaled_value)
            
            Replace = st.selectbox("Replace values of column?", ("No","Yes"),key='scaler')
            if Replace == "Yes":
                df[col_scaled] = scaled_value
                st.write("Added scaled column in dataframe")
                st.write(df.head())
            elif Replace == "No":
                st.write('No values replaced yet')

        st.sidebar.title('Download processed dataset')
        if st.sidebar.checkbox("download file"):
            st.sidebar.markdown(helper.get_table_download_link(df), unsafe_allow_html=True)

        if st.sidebar.button('Credits'):
            st.sidebar.markdown('''

            **Md.Sadab Wasim**

            Get in touch: [Twitter](https://twitter.com/@sadab_wasim)

            Source Code: [Github](https://github.com/mdsadabwasim/reconn)
            ''')


if __name__ == '__main__':
    main()
