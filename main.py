import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
from secrets import G_KEY
import plotly.express as px


def change(row):
    row = str(row).strip()
    row = row.replace(' ', '')
    row = row.replace('+', '')
    try:
        return int(float(row))
    except Exception:
        return np.nan

def cust_with_us(row):
    try:
        if row <= 30:
            return '0-30'
        elif row <= 40:
            return '31-40'
        elif row <= 50:
            return '41-50'
        elif row <= 60:
            return '51-60'
        elif row <= 70:
            return '61-70'
        elif row <= 80:
            return '71-80'
        elif row <= 90:
            return '81-90'
        elif row > 90:
            return '91+'
        else:
            return row
    except Exception:
        return row

def delta(row, df):
    gr = df.groupby('Full Name')['Total Transactions'].sum()
    name = row['Full Name']
    these_day = map(str, [datetime.date.today().year, datetime.date.today().month, datetime.date.today().day][::-1])
    these_day = '-'.join(these_day)
    these_day = datetime.datetime.strptime(these_day, '%d-%m-%Y')
    try:
        diff = int((these_day - row['Signup Date']).days)
        return round(diff / gr[name], 2)
    except Exception:
        return 'Undisclosed'

def changedate(row):
    try:
        row = row.split()[0]
        return pd.to_datetime(row)
    except Exception:
        return np.nan

def age(row):
    y = datetime.date.today().year
    try:
        return abs(int(y - row.year))
    except Exception:
        return row

def spend_cat(row):
    try:
        if row == 0:
            return 'Unknown (0)'
        elif row <= 99.99:
            return '1-99'
        elif row <= 499.99:
            return '100-499'
        elif row <= 999.99:
            return '500-999'
        elif row <= 4999.99:
            return '1000-4999'
        elif row <= 9999:
            return '5000+'
        else:
            return row
    except Exception:
        return row

def freq_cat(row):
    try:
        if row < 10:
            return 'Under 10'
        elif row <= 19:
            return '10-19'
        elif row <= 29:
            return '20-29'
        elif row <= 39:
            return '30-39'
        elif row <= 49:
            return '40-49'
        elif row <= 59:
            return '50-59'
        elif row >= 60:
            return '60+'
        else:
            return row
    except Exception:
        return row

def age_cat(data):
    row = data['Age']
    try:
        if row < 17:
            return 'Undisclosed'
        elif row <= 24:
            return '17-24'
        elif row <= 34:
            return '25-34'
        elif row <= 44:
            return '35-44'
        elif row <= 54:
            return '45-54'
        elif row <= 64:
            return '55-64'
        elif row >= 65:
            return '65+'
        else:
            return row
    except Exception:
        return row

def get_in_time(row):
    try:
        if 'hour' in row:
            return int(row.split()[0]) * 60 + int(row.split()[2])
        else:
            return int(row.split()[0])
    except Exception:
        return np.nan

def locality(row):
    try:
        if row <= 5:
            return 'Local (0-5km)'
        elif row <= 7:
            return 'London destination (5-7km)'
        elif row <= 10:
            return 'London destination (7-10km)'
        elif row <= 300:
            return 'Out of London desitination (10-300km)'
        elif row > 300:
            return 'Out of UK destination (>300km)'
        else:
            return row
    except Exception:
        return row

def date_cleanup(row):
    try:
        row = row.split()[0]
        return pd.to_datetime(row, errors='ignore')
    except Exception:
        return row

def total_sp(row):
    row = str(row).strip()
    try:
        return float(row)
    except:
        return np.nan

# ------------------
# Streamlit App
# ------------------

def main():
    st.set_page_config(page_title="Client Data Analysis", layout="wide")

    st.title("Client Data Analysis")
    st.write("Please upload your CSV file below:")

    # File uploader
    uploaded_file = st.file_uploader("Upload Excel", type=["csv"])

    if uploaded_file is not None:

        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = [
            'First Name', 'Last Name', 'Full Name', 'Email', 'Phone', 'Mobile',
            'Date of Birth', 'Salutation', 'Mailing Postal Code', 'Gender', 'Language',
            'How did you find about us?', 'WhatsApp', 'Subscriber', 'Signup Place',
            'Total Spend', 'Average Spend', 'Total Transactions', 'Signup Date',
            'Last Transaction Date', 'EPOS ID'
        ]
        if not set(required_cols).issubset(df.columns):
            st.error("Invalid columns! Please check your CSV file.")
            return

        # Keep only necessary columns
        df = df[required_cols]

        # Uppercase mailing postal code
        df['Mailing Postal Code'] = df['Mailing Postal Code'].str.upper()

        # Example distance-matrix integration (use your valid API key)
        # Here, we only show how you might call the API in chunks:
        mas = list(df['Mailing Postal Code'].dropna().unique())
        d = {}
        if len(mas) > 0:
            for i in range(int(np.ceil(len(mas) / 25))):
                destinations_chunk = "|".join(mas[25 * i : 25 * (i + 1)])
                url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins=W4 5TF&destinations={destinations_chunk}&units=metric&mode=car&key={G_KEY}"
                response = requests.request("GET", url)
                q = eval(response.text)
                if q.get('status') == 'OK':
                    a = destinations_chunk.split("|")
                    for elem in range(len(q['rows'][0]['elements'])):
                        if q['rows'][0]['elements'][elem]['status'] == 'OK':
                            dist = float(q['rows'][0]['elements'][elem]['distance']['text']
                                        .split()[0].replace(',', ''))
                            time_ = q['rows'][0]['elements'][elem]['duration']['text']
                        else:
                            dist = np.nan
                            time_ = np.nan
                        d[a[elem]] = [dist, time_]

            frame = pd.DataFrame(d).T.reset_index().rename(columns={
                'index': 'Mailing Postal Code',
                0: 'Distance, km',
                1: 'Duration'
            })
            df = pd.merge(df, frame, how='left', on='Mailing Postal Code')
            df['Duration in min'] = df['Duration'].apply(get_in_time)
            df['Locality'] = df['Distance, km'].apply(locality)
        else:
            df['Distance, km'] = np.nan
            df['Duration'] = np.nan
            df['Duration in min'] = np.nan
            df['Locality'] = np.nan

        # Data cleaning / filling
        df['Total Transactions'] = df['Total Transactions'].fillna(0)
        df.loc[df['Total Transactions'] == 0, 'Total Transactions'] = 1
        df['Gender'] = df['Gender'].fillna('Undisclosed')
        df['Language'] = df['Language'].fillna('Undisclosed')
        df['Phone'] = df['Phone'].fillna('nan')
        df['Phone'] = df['Phone'].apply(change)
        df['Mobile'] = df['Mobile'].apply(change)
        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce', dayfirst=True)
        df['Signup Date'] = df['Signup Date'].apply(date_cleanup)
        df['Total Spend'] = df['Total Spend'].apply(total_sp)
        df['Last Transaction Date'] = df['Last Transaction Date'].apply(changedate)
        df['Total Transactions'] = df['Total Transactions'].astype('int')
        df['Subscriber'] = df['Subscriber'].fillna('No')

        # Basic stats
        total_spend = np.sum(df['Total Spend'])
        total_customers = df['Total Transactions'].count()
        total_visits = np.sum(df['Total Transactions'])
        avg_freq = round(total_visits / total_customers, 2)
        avg_spend_per_visit = round(total_spend / total_visits, 2)
        subs = (df['Subscriber'] == 'Yes').sum()
        subs_share = round(subs / df.shape[0] * 100, 2)

        # Additional columns
        df['Age'] = df['Date of Birth'].apply(age)
        df['Spend_category'] = df['Total Spend'].apply(spend_cat)
        df['Frequency'] = df['Total Transactions'].apply(freq_cat)
        df['Age_category'] = df.apply(age_cat, axis=1)

        # Pivot tables
        piv_spend = df.groupby('Spend_category')[['Total Spend', 'Total Transactions']].agg({
            'Total Transactions': ['count', 'sum'],
            'Total Spend': ['sum', 'mean']
        })
        piv_spend.columns = piv_spend.columns.droplevel(0)
        piv_spend.columns = ['n. customer type', 'Total transactions', 'Total spend', 'Average spend']
        piv_spend['n. customer type share (%)'] = round(
            piv_spend['n. customer type'] / piv_spend['n. customer type'].sum() * 100, 2
        )
        piv_spend['Total transaction share (%)'] = round(
            piv_spend['Total transactions'] / piv_spend['Total transactions'].sum() * 100, 2
        )
        piv_spend['Total spend share (%)'] = round(
            piv_spend['Total spend'] / piv_spend['Total spend'].sum() * 100, 2
        )
        piv_spend['Average spend'] = round(piv_spend['Average spend'], 2)
        piv_spend = piv_spend[
            ['n. customer type', 'n. customer type share (%)',
             'Total spend', 'Total spend share (%)',
             'Total transactions', 'Total transaction share (%)', 'Average spend']
        ]
        piv_spend = piv_spend.reindex(index=['Unknown (0)', '1-99', '100-499', '500-999', '1000-4999', '5000+']).fillna(0)

        piv_freq = df.groupby(['Frequency', 'Total Transactions'])[['Total Spend', 'Total Transactions']].agg({
            'Total Spend': [np.sum],
            'Total Transactions': ['count', np.sum]
        })
        piv_freq.columns = piv_freq.columns.droplevel(0)
        piv_freq.columns = ['Total spend sum', 'Count of customer type', 'Total transactions']
        piv_freq['Total spend share (%)'] = round(
            piv_freq['Total spend sum'] / piv_freq['Total spend sum'].sum() * 100, 2
        )
        piv_freq['Total transactions (%)'] = round(
            piv_freq['Total transactions'] / piv_freq['Total transactions'].sum() * 100, 2
        )
        piv_freq = piv_freq[['Count of customer type', 'Total spend sum', 'Total spend share (%)',
                             'Total transactions', 'Total transactions (%)']]
        piv_freq = piv_freq.sort_values(by='Total transactions')

        piv_age = df.groupby('Age_category')[['Total Spend', 'Age']].agg({
            'Total Spend': 'sum',
            'Age': ['count', 'sum']
        })
        piv_age.columns = piv_age.columns.droplevel(0)
        piv_age.columns = ['Total spend', 'Count of age status', 'Total transactions']
        piv_age['Total spend share (%)'] = round(
            piv_age['Total spend'] / piv_age['Total spend'].sum() * 100, 2
        )
        piv_age['Total transactions share (%)'] = round(
            (piv_age['Total transactions'] / piv_age['Total transactions'].sum() * 100).astype('float'), 2
        )
        piv_age = piv_age[['Count of age status', 'Total spend', 'Total spend share (%)',
                           'Total transactions', 'Total transactions share (%)']]
        piv_age = piv_age.reindex(index=['Undisclosed', '17-24', '25-34', '35-44', '45-54', '55-64', '65+']).fillna(0)

        piv_gender = df.groupby('Gender')[['Total Spend', 'Total Transactions']].agg({
            'Total Transactions': ['count', 'sum', 'mean'],
            'Total Spend': ['sum', 'mean']
        })
        piv_gender.columns = piv_gender.columns.droplevel(0)
        piv_gender.columns = ['n. customer type', 'Total transactions sum', 'Average transactions',
                              'Total spend sum', 'Average spend']
        piv_gender['Average spend'] = round(piv_gender['Average spend'], 2)
        piv_gender['Average transactions'] = round(piv_gender['Average transactions'], 2)
        piv_gender['n. customer type share (%)'] = round(
            piv_gender['n. customer type'] / piv_gender['n. customer type'].sum() * 100, 2
        )
        piv_gender['Transaction count share (%)'] = round(
            piv_gender['Total transactions sum'] / piv_gender['Total transactions sum'].sum() * 100, 2
        )
        piv_gender['Total spend share (%)'] = round(
            piv_gender['Total spend sum'] / piv_gender['Total spend sum'].sum() * 100, 2
        )
        piv_gender = piv_gender[
            ['n. customer type', 'n. customer type share (%)',
             'Total spend sum', 'Total spend share (%)', 'Average spend',
             'Total transactions sum', 'Transaction count share (%)', 'Average transactions']
        ]

        piv_lang = df.groupby('Language')[['Total Spend', 'Total Transactions']].agg({
            'Total Transactions': ['count', 'sum'],
            'Total Spend': ['sum', 'mean']
        })
        piv_lang.columns = piv_lang.columns.droplevel(0)
        piv_lang.columns = ['n. customer type', 'Total transactions sum', 'Total spend sum', 'Average spend']
        piv_lang['n. customer type share (%)'] = round(
            piv_lang['n. customer type'] / piv_lang['n. customer type'].sum() * 100, 2
        )
        piv_lang['Transaction count share (%)'] = round(
            piv_lang['Total transactions sum'] / piv_lang['Total transactions sum'].sum() * 100, 2
        )
        piv_lang['Total spend share (%)'] = round(
            piv_lang['Total spend sum'] / piv_lang['Total spend sum'].sum() * 100, 2
        )
        piv_lang = piv_lang[
            ['n. customer type', 'n. customer type share (%)',
             'Total spend sum', 'Total spend share (%)',
             'Total transactions sum', 'Transaction count share (%)', 'Average spend']
        ]

        how_found = round(df['How did you find about us?'].value_counts(normalize=True) * 100, 2).to_frame().reset_index()
        how_found.columns = ['How did you find about us?', 'Share of category']
        how_found = how_found.sort_values(by='Share of category', ascending=False)

        # Local / repeat customers
        if 'Locality' not in df.columns:
            df['Locality'] = np.nan
        local = (
            df['Locality']
            .value_counts()
            .to_frame()
            .rename(columns={'Locality': 'Amount of customers'})
            .reindex(index=[
                'Local (0-5km)', 'London destination (5-7km)',
                'London destination (7-10km)', 'Out of London desitination (10-300km)',
                'Out of UK destination (>300km)'
            ])
            .fillna(0)
        )

        df['Delta days'] = df.apply(lambda x: (x['Last Transaction Date'] - x['Signup Date']).days, axis=1)
        df['Freq'] = df.apply(lambda x: delta(x, df), axis=1)
        df['Repeat customers'] = df['Freq'].apply(cust_with_us)
        
        piv_freq_cat = (
            df['Repeat customers']
            .value_counts()
            .to_frame()
            .reindex(index=['Undisclosed', '0-30', '31-40', '41-50',
                            '51-60', '61-70', '71-80', '81-90', '91+'])
            .fillna(0)
        )

        # -------------------------------
        # Displaying results on Streamlit
        # -------------------------------

        repeat_customers_percentage = round((df['Repeat customers'].isin(['0-30', '31-40', '41-50', '51-60', '61-70', '71-80']).mean()) * 100, 2)

        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6, col7 = st.columns(4)
        
        col1.metric("Total Spend", f"{round(total_spend,2):,} £")
        col2.metric("Total Customers", f"{total_customers}")
        col3.metric("Total Visits", f"{int(total_visits)}")
        col4.metric("Subscribers", f"{subs} ({subs_share}%)")
        col5.metric("Average Freq", f"{avg_freq:.2f}")
        col6.metric("Avg Spend/Visit", f"{avg_spend_per_visit:,.2f} £")
        col7.metric('Repeat Customers (<91 days)', f"{repeat_customers_percentage}%")

        # -------------------------------
        # Charts (6 examples)
        # -------------------------------
        st.subheader("Charts")

        col8, col9 = st.columns(2)

        # 1. Pie chart: n. customer type by Spend_category
        with col8:
            fig_pie = px.pie(
                piv_spend.reset_index(),
                names='Spend_category',
                values='n. customer type',
                title="n. customer type by Spend category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # 2. Bar chart: Total spend by Spend_category
            fig_bar_total_spend = px.bar(
                piv_spend.reset_index(),
                x='Spend_category',
                y='Total spend',
                color='Spend_category',
                title="Total spend by Spend category",
                text='Total spend'
            )
            fig_bar_total_spend.update_traces(textposition='outside')
            st.plotly_chart(fig_bar_total_spend, use_container_width=True)

        # 3. Bar chart: Total transactions by Spend_category
        with col9:
            fig_bar_total_transactions = px.bar(
                piv_spend.reset_index(),
                x='Spend_category',
                y='Total transactions',
                color='Spend_category',
                title="Total transactions by Spend category",
                text='Total transactions'
            )
            fig_bar_total_transactions.update_traces(textposition='outside')
            st.plotly_chart(fig_bar_total_transactions, use_container_width=True)

            # 4. Line plot: Average spend by Spend_category
            fig_line_avg_spend = px.line(
                piv_spend.reset_index(),
                x='Spend_category',
                y='Average spend',
                title="Average spend by Spend category",
                markers=True,
                text='Average spend'
            )
            fig_line_avg_spend.update_traces(textposition='top center')
            st.plotly_chart(fig_line_avg_spend, use_container_width=True)

        # 2. Frequency Pivot
        piv_freq = piv_freq.reset_index()

        # Decide on pie vs. bar based on the chosen column
        col10 = st.columns(1)[0]
        col11, col12 = st.columns(2)

        piv_freq['Total Transactions'] = piv_freq['Total Transactions'].astype('object')
        piv_freq['Total Transactions'] = piv_freq['Total Transactions'].round(2)

        with col10:
            fig = px.bar(
                piv_freq,
                x="Total Transactions",
                y="Total spend sum",
                color="Frequency",
                text="Total spend sum",
                title="Total Transactions vs Total Spend Sum",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col11:
            fig = px.pie(
                piv_freq,
                names="Frequency",
                values="Count of customer type",
                title=f"'Count of customer type' by Frequency",
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col12:
            fig = px.pie(
                piv_freq,
                names="Frequency",
                values="Total transactions",
                title=f"'Total transactions' by Frequency",
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3. Age Distribution
        fig_age = px.bar(
            piv_age.reset_index(),
            x='Age_category',
            y='Count of age status',
            color='Age_category',
            title="Age Category - Count of Customers",
            text='Count of age status'
        )
        fig_age.update_traces(textposition='outside')
        st.plotly_chart(fig_age, use_container_width=True)

        # 4. Gender Pivot
        fig_gender = px.bar(
            piv_gender.reset_index(),
            x='Gender',
            y='n. customer type',
            color='Gender',
            title="Gender - # of Customers",
            text='n. customer type'
        )
        fig_gender.update_traces(textposition='outside')
        st.plotly_chart(fig_gender, use_container_width=True)

        # 5. Language Pivot
        fig_lang = px.bar(
            piv_lang.reset_index(),
            x='Language',
            y='n. customer type',
            color='Language',
            title="Language - # of Customers",
            text='n. customer type'
        )
        fig_lang.update_traces(textposition='outside')
        st.plotly_chart(fig_lang, use_container_width=True)

        # 6. "How Found" Distribution
        how_found_df = how_found.rename(columns={'How did you find about us?': 'Channel', 'Share of category': 'Share (%)'})
        fig_how_found = px.pie(
            how_found_df,
            names='Channel',
            values='Share (%)',
            title="How Did You Find About Us? (Share %)"
        )
        st.plotly_chart(fig_how_found, use_container_width=True)

        # -------------------------------
        # Display final table & download
        # -------------------------------
        st.subheader("Processed Data Table")
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()