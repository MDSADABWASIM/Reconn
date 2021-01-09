import base64
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

labelEncoder = LabelEncoder()
standartScaler = StandardScaler()
minMaxScaler = MinMaxScaler()

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="dataframe.csv">Click to download</a>'
