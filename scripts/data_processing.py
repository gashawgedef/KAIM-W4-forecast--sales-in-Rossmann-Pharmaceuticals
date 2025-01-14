# # data_preprocessing.py
# import pandas as pd
# import logging

# # Set up logging
# logging.basicConfig(filename="data_preprocessing.log", level=logging.INFO)


# def preprocess_data(train_df, store_df):
#     train_df["Date"] = pd.to_datetime(train_df["Date"])
#     train_df["DayOfWeek"] = train_df["Date"].dt.dayofweek
#     train_df["IsWeekend"] = (train_df["DayOfWeek"] >= 5).astype(int)

#     # Merge with store data to include additional features
#     train_df = train_df.merge(store_df, on="Store", how="left")
#     train_df.fillna(0, inplace=True)

#     logging.info("Data preprocessing complete.")
#     return train_df


# if __name__ == "__main__":
#     data_path = "C:\\Users\\HP\\Desktop\\10 academy\\KAIM-W4-forecast--sales-in-Rossmann-Pharmaceuticals\\data\\"
#     store_df = pd.read_csv(data_path + "store.csv")
#     train_df = pd.read_csv(data_path + "train.csv")

#     preprocessed_train_df = preprocess_data(train_df, store_df)
#     print("Data preprocessing complete.")
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename="data_preprocessing.log", level=logging.INFO)


def preprocess_data(train_df, store_df):
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df["DayOfWeek"] = train_df["Date"].dt.dayofweek
    train_df["IsWeekend"] = (train_df["DayOfWeek"] >= 5).astype(int)

    # Merge with store data to include additional features
    train_df = train_df.merge(store_df, on="Store", how="left")
    train_df.fillna(0, inplace=True)

    logging.info("Data preprocessing complete.")
    return train_df


if __name__ == "__main__":
    data_path = "C:\\Users\\HP\\Desktop\\10 academy\\KAIM-W4-forecast--sales-in-Rossmann-Pharmaceuticals\\data\\"
    output_path = data_path + "preprocessed_train.csv"  # Define output file path

    store_df = pd.read_csv(data_path + "store.csv")
    train_df = pd.read_csv(data_path + "train.csv")

    # Preprocess the data
    preprocessed_train_df = preprocess_data(train_df, store_df)

    # Save the preprocessed data to a new CSV file
    preprocessed_train_df.to_csv(output_path, index=False)
    logging.info(f"Preprocessed data saved to {output_path}")
    print(f"Data preprocessing complete. File saved to: {output_path}")
