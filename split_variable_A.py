import pandas as pd
fs = 500
splitting = []

trial_file = "data/1/trial_data.xlsx"
df = pd.read_excel(trial_file)
df_filtered = df[df["variable_A"] == 1]
print(df_filtered)

trial_start = df_filtered["trial_start"].values[0]
print(trial_start)
init_position = int(trial_start * fs)
print(init_position)

trial_end = df_filtered["trial_end"].values[0]
print(trial_end)
end_position = int(trial_end * fs)
print(end_position)

for i, row in df_filtered.iterrows():
    init_position = int(row["trial_start"] * fs)
    end_position = int(row["trial_end"] * fs)
    splitting.append([init_position, end_position])

print(splitting)





