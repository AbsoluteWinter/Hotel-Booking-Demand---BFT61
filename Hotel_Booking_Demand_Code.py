#####################################################################################
########################### HOTEL BOOKING DEMAND RAW CODE ###########################
#####################################################################################
# Install extra packages
%pip install -U -q absdataset # Download dataset for this project
%pip install -U -q rich[jupyter] # Print formating
%pip install -U -q pycountry # Convert name
#####################################################################################
# Load libraries
import absdataset as abd # Dataset
from rich.jupyter import print # Print format
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
import plotly.express as px
import warnings
#####################################################################################
warnings.filterwarnings("ignore") # No warnings
# Theme setting
THEME = {
    "Title color": "blue on white",
    "Main color": "cyan",
    "Background color": "white",
    "sns": "Blues_r",
    "sns1": "Blues"
}
# Quick variables
MONTH = [
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
    "November", "December"
]
#####################################################################################
pd.set_option("display.max_columns", None) # Show all cols
data = abd.load_data(abd.CSV_DATA["hotel"]) # pandas.read_csv wrapper
data
#####################################################################################
# First look - Overview
_mv = data.isna().sum().sum() # Total missing values
_to = data.shape[0]*data.shape[1] # Total observations
print(f"""\
[b]Data Overview:[/b]
- Rows: {data.shape[0]:,.0f}
- Columns: {data.shape[1]:,.0f}
- Missing values: {_mv:,.0f} 
- Missing values (%): {_mv/_to*100:,.2f} 
""")
#####################################################################################
data.info() # Quick info
#####################################################################################
# Quick describe
data[data.select_dtypes(exclude="object").columns].describe().T.\
            style.background_gradient(axis=0, cmap=THEME["sns1"])
#####################################################################################
# Filter column with missing values and count it
print(data.isna().sum().\
      sort_values(ascending=False)[data.isna().sum()!=0])
# In percentage
_ROUND_NUM = 5 # Round up setting
print(((data.isna().sum()/len(data)).\
      round(_ROUND_NUM)*100).sort_values(ascending=False)\
      [(data.isna().sum()/len(data))*100!=0])
#####################################################################################
plt.figure(figsize=(20,6))
plt.title(
    "Missing values Heatmap\n",
    fontweight="bold",
    fontsize=18
)
sns.heatmap(data.isna(), cmap=THEME["sns"])
plt.show()
#####################################################################################
# Replace missing value in country with mode
data.country.fillna(data.country.mode()[0], inplace=True)
# Replace missing value in children with mode
data.children.fillna(data.children.mode()[0], inplace=True)
# Replace missing value in agent with 0
data.agent.fillna(0, inplace=True)
# Replace missing value in company with 0
data.company.fillna(0, inplace=True)
#####################################################################################
data.meal.replace("Undefined", "SC", inplace=True)
#####################################################################################
# Remove unmeaningful data
data = data[data.adults != 0] # No adults per reservation
data = data[data.adr > 0.00] # No Average Daily Rate
#####################################################################################
# Convert to datetime
data.reservation_status_date = pd.to_datetime(data.reservation_status_date)
#####################################################################################
# Add full arrival date and convert into datetime
def tmonth_to_num(month:str):
  """Convert month (text form) to number"""
  for i, x in enumerate(MONTH):
    if month.startswith(x):
      if i < 9:
        return f"0{i+1}"
      else:
        return f"{i+1}"
  raise SystemExit("Invalid month")
data["arrival_date"] = data.arrival_date_year.astype(str) + "-" +\
                       data.arrival_date_month.astype(str).\
                       apply(tmonth_to_num) + "-" +\
                       data.arrival_date_day_of_month.astype(str)
data.arrival_date = pd.to_datetime(data.arrival_date)
#####################################################################################
# Add total nights stayed column into dataframe
data["total_nights"] = data.stays_in_weekend_nights +\
                              data.stays_in_week_nights
#####################################################################################
# Add total guest
data.children = data.children.astype(int) # Convert to int
data["total_guest"] = data.adults + data.children
# Add price per person
data["price_per_guest"] = data.adr / data.total_guest
#####################################################################################
# Group data by month
monthly_data = data.\
groupby(pd.Grouper(key="arrival_date", axis=0, freq="M")).sum()
#####################################################################################
_mv = data.isna().sum().sum()
print(f"""\
After cleaning, there is/are {_mv} missing value(s)
and {data.shape[0]:,.0f} rows.""")
#####################################################################################
def sns_show_values(axs, orient="v", space=.01, fontsize=14, roundup=True):
  def _single(ax):
    if orient == "v":
      for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
        if roundup:
          value = f"{p.get_height():.0f}"
        else:
          value = f"{p.get_height():.1f}"
        ax.text(_x, _y, value, ha="center", fontsize=fontsize) 
    elif orient == "h":
      for p in ax.patches:
        _x = p.get_x() + p.get_width() + float(space)
        _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
        if roundup:
          value = f"{p.get_width():.0f}"
        else:
          value = f"{p.get_width():.1f}"
        ax.text(_x, _y, value, ha="left", fontsize=fontsize)
  if isinstance(axs, np.ndarray):
    for idx, ax in np.ndenumerate(axs):
      _single(ax)
  else:
    _single(axs)
#####################################################################################
def get_country_name(short_name:str):
  """Quick function convert country name
  from ISO 3166-1 format to normal name"""
  if short_name is None:
    return "Unknown"
  for country in list(pycountry.countries):
    if country.alpha_3.upper().startswith(short_name):
      return country.name
  return "Unknown"
#####################################################################################
plt.figure(figsize=(20, 8))
# Fig 01
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
temp = sns.countplot(
    data=data,
    x="hotel",
    hue="is_canceled",
    palette=THEME["sns1"],
)
sns_show_values(temp, roundup=True)
plt.title(
    "Hotel Types and Reservation Count\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Hotel Type", fontsize=14)
plt.ylabel("Reservation Count", fontsize=14)
plt.legend(
    ["No", "Yes"],
    title="Is canceled",
    shadow=True,
    fontsize=12,
    title_fontsize=14,
    loc="upper left"
)
# Fig 02
plt.subplot(1, 2, 2)
plt.pie(
    data.hotel.value_counts(),
    autopct="%.1f%%",
    colors=sns.color_palette(THEME["sns"]),
    textprops={"fontsize": 14, "color": "white", "fontweight": "bold"}
)
plt.title(
    "Hotel Types and Reservation Count (%)\n",
    fontweight="bold", fontsize=16
)
plt.legend(
    ["City Hotel", "Resort Hotel"],
    title="Hotel Type", shadow=True,
    loc="lower right",
    fontsize=12,
    title_fontsize=14
)
plt.show()
#####################################################################################
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
temp = sns.countplot(
    data=data, 
    x="arrival_date_month", 
    hue="hotel",
    order=MONTH,
    palette=THEME["sns1"]
)
sns_show_values(temp, fontsize=10, roundup=True)
plt.title(
    "Monthly Reservation Count by Hotel Type\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Month", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.legend(
    title="Hotel Type",
    shadow=True,
    loc="best",
    fontsize=11,
    title_fontsize=12
)
plt.show()
#####################################################################################
plt.figure(figsize=(14, 8))
sns.set_style("darkgrid")
sns.lineplot(
    x="arrival_date_month",
    y="total_nights",
    hue="arrival_date_year",
    data=data,
    palette="crest",
)
plt.title(
    "Total stay throughout the year\n",
    fontweight="bold",
    fontsize=16
)
plt.xlabel("Month", fontsize = 14)
plt.ylabel("Total night stayed", fontsize = 14)
plt.legend(
    title="Year",
    shadow=True,
    loc="best",
    fontsize=12,
    title_fontsize=14
)
plt.show()
#####################################################################################
# Make separated country data
country_data = pd.DataFrame(
    data.loc[data["is_canceled"] == 0]["country"].value_counts()
)
country_data.rename(
    columns={"country": "Number of Guests"},
    inplace=True
)
total_guests = country_data["Number of Guests"].sum()
country_data["Guests in %"] = round(country_data["Number of Guests"] \
                                    / total_guests * 100, 2)
country_data["country"] = country_data.index
country_data_f = country_data.copy() # Make copy
# Get full country name and filter out
country_data_f["country_f"] = country_data_f["country"].\
                                  apply(get_country_name)
country_data_f.\
      loc[country_data_f["Guests in %"] < 2, "country_f"] = "Other"
# Pie chart
temp = px.pie(
    country_data_f,
    values="Number of Guests",
    names="country_f",
    title="Home country of guests",
    color_discrete_sequence=px.colors.sequential.Blues_r,
    hole=.3,
    labels={"country_f": "Country"}
)
temp.update_traces(
    textposition="inside",
    textinfo="value+percent+label"
)
temp.update_layout(
    font=dict(size=16),
    annotations=[
        dict(text="Country", x=0.5, y=0.5,
             font_size=20, showarrow=False)
    ]
)
temp.show()
#####################################################################################
# Map version
temp = px.choropleth(
    country_data,
    locations=country_data.index,
    color=country_data["Guests in %"], 
    hover_name=country_data.country.apply(get_country_name), 
    color_continuous_scale=px.colors.sequential.Blues,
    title="Home country of guests"
)
temp.update_layout(font=dict(size=16))
temp.show()
#####################################################################################
plt.figure(figsize=(20, 8))
# Fig 01
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
temp = sns.countplot(
    data=data,
    x="hotel",
    hue="meal",
    palette=THEME["sns1"],
)
sns_show_values(temp, roundup=True)
plt.title(
    "Types of Meal Booked\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Hotel Type", fontsize=14)
plt.ylabel("Meal Count", fontsize=14)
plt.legend(
    ["Bed & Breakfast", "Full board", "Half board", "No meal package"],
    title="Meal type",
    shadow=True,
    fontsize=12,
    title_fontsize=14,
    loc="upper left"
)
# Fig 02
plt.subplot(1, 2, 2)
plt.pie(
    data.meal.value_counts(),
    autopct="%.1f%%",
    colors=sns.color_palette(THEME["sns1"]),
    textprops={"fontsize": 14, "color": "black", "fontweight": "bold"}
)
plt.title(
    "Types of Meal Booked (%)\n",
    fontweight="bold", fontsize=16
)
plt.legend(
    ["Bed & Breakfast", "Half board", "No meal package", "Full board"],
    title="Meal type", shadow=True,
    loc="lower right",
    fontsize=12,
    title_fontsize=14
)
plt.show()
#####################################################################################
# Calculate
total_cancel = data["is_canceled"].sum()
rh_cancel = data.\
    loc[data["hotel"] == "Resort Hotel"]["is_canceled"].sum()
ch_cancel = data.\
    loc[data["hotel"] == "City Hotel"]["is_canceled"].sum()
# Percentage form:
p_cancel = total_cancel / data.shape[0] * 100
rh_p_cancel = rh_cancel /\
    data.loc[data["hotel"] == "Resort Hotel"].shape[0] * 100
ch_p_cancel = ch_cancel /\
    data.loc[data["hotel"] == "City Hotel"].shape[0] * 100
# Output
print(f"""\
[b]Total bookings canceled: {total_cancel:,} ({p_cancel:.0f}%)[/]
[b]Resort hotel[/] bookings canceled: {rh_cancel:,} ({rh_p_cancel:.0f}%)
[b]City hotel[/] bookings canceled: {ch_cancel:,} ({ch_p_cancel:.0f}%)
""")
#####################################################################################
# Make separated data
# Resort Hotel
rh_book_per_month = data.loc[(data["hotel"] == "Resort Hotel")].\
                  groupby("arrival_date_month")["hotel"].count()
rh_cancel_per_month = data.loc[(data["hotel"] == "Resort Hotel")].\
                  groupby("arrival_date_month")["is_canceled"].sum()
rh_cancel_dat = pd.DataFrame({
    "Hotel": "Resort Hotel",
    "Month": list(rh_book_per_month.index),
    "Bookings": list(rh_book_per_month.values),
    "Cancelations": list(rh_cancel_per_month.values)
})
# City Hotel
ch_book_per_month = data.loc[(data["hotel"] == "City Hotel")].\
                  groupby("arrival_date_month")["hotel"].count()
ch_cancel_per_month = data.loc[(data["hotel"] == "City Hotel")].\
                  groupby("arrival_date_month")["is_canceled"].sum()
ch_cancel_dat = pd.DataFrame({
    "Hotel": "City Hotel",
    "Month": list(ch_book_per_month.index),
    "Bookings": list(ch_book_per_month.values),
    "Cancelations": list(ch_cancel_per_month.values)
})
# Concat data
full_cancel_dat = pd.concat([rh_cancel_dat, ch_cancel_dat],
                             ignore_index=True)
full_cancel_dat["cancel_percent"] = full_cancel_dat["Cancelations"] \
                                  / full_cancel_dat["Bookings"] * 100
# Order by month
full_cancel_dat.Month = pd.Categorical(
    full_cancel_dat.Month,
    categories=MONTH,
    ordered=True
)
# Bar chart
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.barplot(
    x="Month",
    y="cancel_percent",
    hue="Hotel",
    hue_order = ["City Hotel", "Resort Hotel"],
    data=full_cancel_dat,
    palette=THEME["sns1"]
)
plt.title(
    "Cancelations per month\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Month", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Cancelations (%)", fontsize=14)
plt.legend(
    title="Hotel Type",
    shadow=True,
    loc="upper left",
    fontsize=11,
    title_fontsize=12
)
plt.show()
#####################################################################################
# Resort Hotel
rh = data.loc[(data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 0)]
# City Hotel
ch = data.loc[(data["hotel"] == "City Hotel") & (data["is_canceled"] == 0)]
#####################################################################################
# Check average price
_rh_pmean = rh.price_per_guest.mean()
_ch_pmean = ch.price_per_guest.mean()
print(f"""\
[b]The average prices from all non-canceled bookings:[/]
Resort Hotel: EUR {_rh_pmean:,.2f} per night per person.
City Hotel: EUR {_ch_pmean:,.2f} per night per person.
""")
#####################################################################################
# Make separated data
# Take out non-canceled booking
guest_dat = data.loc[data.is_canceled == 0]
room_prices = guest_dat[
    ["hotel", "reserved_room_type", "price_per_guest"]
].sort_values("reserved_room_type")
# Boxplot
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.boxplot(
    x="reserved_room_type",
    y="price_per_guest",
    hue="hotel",
    data=room_prices, 
    hue_order=["City Hotel", "Resort Hotel"],
    fliersize=0,
    palette=THEME["sns1"]
)
plt.title(
    "Price of room types per night per person\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Room type", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.legend(
    title="Hotel Type", shadow=True,
    loc="upper right",
    fontsize=11, title_fontsize=12
)
plt.ylim(0, 160)
plt.show()
#####################################################################################
# Make separated data
room_prices_m = guest_dat[
    ["hotel", "arrival_date_month", "price_per_guest"]
].sort_values("arrival_date_month")
room_prices_m["arrival_date_month"] = pd.Categorical(
    room_prices_m["arrival_date_month"],
    categories=MONTH, ordered=True
)
# Line graph with standard deviation
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.lineplot(
    x="arrival_date_month",
    y="price_per_guest",
    hue="hotel", 
    data=room_prices_m, 
    hue_order = ["City Hotel", "Resort Hotel"],
    ci="sd",
    size="hotel",
    sizes=(2.5, 2.5),
    palette=THEME["sns1"]
)
plt.title(
    "Room price per night per person over the year\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Month", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Price", fontsize=14)
plt.legend(
    title="Hotel Type",
    shadow=True, loc="upper right",
    fontsize=11, title_fontsize=12
)
plt.show()
#####################################################################################
# Make separated data
# Resort Hotel
rh_guests_m = rh.groupby("arrival_date_month")["hotel"].count()
rh_guest_dat = pd.DataFrame({
    "month": list(rh_guests_m.index),
    "hotel": "Resort Hotel", 
    "guests": list(rh_guests_m.values)
})
# City Hotel
ch_guests_m = ch.groupby("arrival_date_month")["hotel"].count()
ch_guest_dat = pd.DataFrame({
    "month": list(ch_guests_m.index),
    "hotel": "City Hotel", 
    "guests": list(ch_guests_m.values)
})
# Concat data
full_guest_dat = pd.concat(
    [rh_guest_dat, ch_guest_dat],
    ignore_index=True
)
# Order by month
full_guest_dat.month = pd.Categorical(
    full_guest_dat.month,
    categories=MONTH, ordered=True
)
# Normalize data due to July & August has data of 3 years
full_guest_dat.loc[
    (full_guest_dat["month"] == "July") |\
    (full_guest_dat["month"] == "August"), "guests"
] /= 3
full_guest_dat.loc[
    ~((full_guest_dat["month"] == "July") |\
      (full_guest_dat["month"] == "August")), "guests"
] /= 2
# Line Graph
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.lineplot(
    x="month", 
    y="guests",
    hue="hotel",
    data=full_guest_dat, 
    hue_order=["City Hotel", "Resort Hotel"],
    size="hotel",
    sizes=(4, 4),
    palette=THEME["sns1"]
)
plt.title(
    "Average number of hotel guests per month\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Month", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Number of guests", fontsize=14)
plt.legend(
    title="Hotel Type",
    shadow=True, loc="upper right",
    fontsize=11, title_fontsize=12
)
plt.show()
#####################################################################################
# Make separated data
# Resort Hotel
num_night_rh = list(rh.total_nights.value_counts().index)
num_book_rh = list(rh.total_nights.value_counts())
p_book_rh = rh.total_nights.value_counts() /\
            sum(num_book_rh) * 100 # Convert to percentage
rh_night = pd.DataFrame({
    "hotel": "Resort Hotel",
    "num_nights": num_night_rh,
    "p_num_bookings": p_book_rh
})
# City Hotel
num_night_ch = list(ch.total_nights.value_counts().index)
num_book_ch = list(ch.total_nights.value_counts())
p_book_ch = ch.total_nights.value_counts() /\
            sum(num_book_ch) * 100 # Convert to percentage
ch_night = pd.DataFrame({
    "hotel": "City Hotel",
    "num_nights": num_night_ch,
    "p_num_bookings": p_book_ch
})
# Concat data
night_dat = pd.concat(
    [rh_night, ch_night],
    ignore_index=True
)
# Bar chart
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
sns.barplot(
    x="num_nights",
    y="p_num_bookings",
    hue="hotel",
    data=night_dat,
    hue_order = ["City Hotel", "Resort Hotel"],
    palette=THEME["sns1"]
)
plt.title(
    "Length of stay\n",
    fontweight="bold", fontsize=16
)
plt.xlabel("Number of nights", fontsize=14)
plt.ylabel("Guests (%)", fontsize=14)
plt.legend(loc="upper right")
plt.xlim(-1,22)
plt.legend(
    title="Hotel Type",
    shadow=True,
    loc="upper right",
    fontsize=11,
    title_fontsize=12
)
plt.show()
#####################################################################################
_m_night_rh = sum(list((rh_night.num_nights *\
                           (rh_night.p_num_bookings/100)).values))
_max_rh = rh_night.num_nights.max()
_m_night_ch = sum(list((ch_night.num_nights *\
                           (ch_night.p_num_bookings/100)).values))
_max_ch = ch_night.num_nights.max()
print(f"""\
[b] City Hotel:\t| Resort Hotel:[/]
- avg: {_m_night_ch:.2f}\t[b]|[/] - avg: {_m_night_rh:.2f}
- max: {_max_ch}\t[b]|[/] - max: {_max_rh}
""")
#####################################################################################
# Make separated data
segments = data["market_segment"].value_counts()
# Pie chart
temp = px.pie(
  segments,
  values=segments.values,
  names=segments.index,
  title="Bookings per market segment",
  color_discrete_sequence=px.colors.\
                    sequential.Blues_r,
)
temp.update_traces(rotation=-90,
                   textinfo="value+percent+label")
temp.update_layout(font=dict(size=16))
temp.show()
#####################################################################################
temp = px.pie(
    data, 
    values=data.customer_type.value_counts(), 
    names=data.customer_type.unique(), 
    title="Customer segment of bookings",
    color_discrete_sequence=px.colors.\
                    sequential.Blues_r
)
temp.update_traces(rotation=180,
                   textinfo="value+percent+label",
                   pull=[0.1, 0, 0, 0])
temp.update_layout(font=dict(size=16))
temp.show()
#####################################################################################
######################################## END ########################################
#####################################################################################