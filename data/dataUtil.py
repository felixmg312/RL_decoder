from datasets import load_dataset
import datasets
data = load_dataset("gigaword",split = 'train[:200000]')
train_data, test_data = data.train_test_split(test_size=0.1).values()
small_dataset = datasets.DatasetDict({'train':train_data,'validation':test_data})
whole_dataset = load_dataset('gigaword')