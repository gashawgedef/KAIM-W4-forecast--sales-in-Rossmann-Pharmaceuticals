import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename="data_loading.log", level=logging.INFO)


def load_data(data_path):
    try:
        store_df = pd.read_csv(data_path + "store.csv")
        train_df = pd.read_csv(data_path + "train.csv")
        test_df = pd.read_csv(data_path + "test.csv")

        logging.info("Datasets loaded successfully")
        return store_df, train_df, test_df
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return None, None, None  # Return None to handle error case in the caller


if __name__ == "__main__":
    data_path = "C:\\Users\\HP\\Desktop\\10 academy\\KAIM-W4-forecast--sales-in-Rossmann-Pharmaceuticals\\data\\"
    store_df, train_df, test_df = load_data(data_path)

    # Check if the datasets were loaded successfully
    if store_df is not None and train_df is not None and test_df is not None:
        print("Store, Train, and Test data loaded.")
    else:
        print("Error loading data. Please check the log file for details.")
