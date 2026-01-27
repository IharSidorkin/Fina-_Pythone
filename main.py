from masz.Data import DataProcessor
from masz.modelm import ModelManager

def main():
    # 1. Data Stage
    processor = DataProcessor()
    processor.load_data()
    processor.exploratory_analysis()
    processor.preprocess_data()

    # 2. Model Stage
    manager = ModelManager(
        processor.X_train,
        processor.X_test,
        processor.y_train,
        processor.y_test
    )
    manager.train_and_compare()
    manager.evaluate()

if __name__ == "__main__":
    main()