from src.data_loader import load_data, clean_data, classify_ctr
from src.eda import plot_distributions, plot_correlations
from src.kpi_calculator import calculate_kpis
from src.model import train_ctr_classifier

# === File path to your dataset ===
file_path = r"data/Dataset_Ads.csv"  # Use relative path if possible

# === Step 1: Load the data ===
df = load_data(file_path)

if df is not None:
    print(f"✅ Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("👉 Columns in file:", df.columns.tolist())
    print("🔍 Sample rows:\n", df.head())

    # === Step 2: Clean the data ===
    try:
        df = clean_data(df)
        print("🧹 Data cleaned successfully.")
    except KeyError as e:
        print(f"❌ Cleaning failed. Missing column: {e}")
        exit()
    except Exception as e:
        print(f"❌ Unexpected error during cleaning: {e}")
        exit()

    # === Step 3: Perform EDA ===
    plot_distributions(df)
    plot_correlations(df)

    # === Step 4: Calculate KPIs ===
    df = calculate_kpis(df)
    print("📈 Sample KPIs:")
    print(df[["CTR", "CPC", "CPA", "ConversionRate"]].head())

    # === Step 5: Classify CTR ===
    df = classify_ctr(df, threshold=0.05)
    print("🏷️ CTR Classification:")
    print(df[["CTR", "High_CTR"]].head())

    # === Step 6: Train model ===
    model, report = train_ctr_classifier(df)
    print("🤖 Model Training Complete.")
    print("📊 Classification Report:\n", report)

else:
    print("❌ Data loading failed. Please check the file path or content.")
