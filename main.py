from data_import import load_all_csvs
from pandasgui import show

def main():
    print("\n========== LOADING CSV FILES ==========")
    data_frames = load_all_csvs()

    print("\n========== LOADED DATAFRAMES ==========")
    for key in data_frames:
        print(f"✔ Loaded: {key} ({len(data_frames[key])} rows)")

    print("\n========== DONE ==========")

    # Uncomment this to open GUI
    # show(**data_frames)

if __name__ == "__main__":
    main()
