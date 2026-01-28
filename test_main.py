import unittest
import pandas as pd
from masz.Data import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DataProcessor()

        # sztuczny zbiór danych do testów
        self.processor.df = pd.DataFrame({
            'sex': ['male', 'female', 'female', 'male'],
            'age': ['20s', '30s', '40s', '50s'],
            'infection_case': ['contact', 'overseas', 'contact', 'unknown'],
            'state': ['released', 'deceased', 'released', 'deceased']
        })

    def test_clean_age(self):
        df = pd.DataFrame({'age': ['50s', '20s', 'unknown']})
        cleaned_df = self.processor._clean_age(df)

        self.assertEqual(cleaned_df['age'][0], 50)
        self.assertEqual(cleaned_df['age'][1], 20)
        self.assertTrue(pd.isna(cleaned_df['age'][2]))

    def test_preprocess_creates_train_test_sets(self):
        self.processor.preprocess_data()

        self.assertIsNotNone(self.processor.X_train)
        self.assertIsNotNone(self.processor.X_test)
        self.assertIsNotNone(self.processor.y_train)
        self.assertIsNotNone(self.processor.y_test)

        self.assertGreater(len(self.processor.X_train), 0)
        self.assertGreater(len(self.processor.X_test), 0)


if __name__ == '__main__':
    unittest.main()