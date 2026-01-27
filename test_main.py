import unittest
import pandas as pd
from main import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()

    def test_clean_age(self):
        """Test if '50s' correctly becomes 50"""
        df = pd.DataFrame({'age': ['50s', '20s', 'unknown']})
        cleaned_df = self.processor.clean_age(df)
        self.assertEqual(cleaned_df['age'][0], 50)
        self.assertEqual(cleaned_df['age'][1], 20)
        self.assertTrue(pd.isna(cleaned_df['age'][2]))

if __name__ == '__main__':
    unittest.main()