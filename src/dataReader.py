import datasets
from datasets import load_dataset


class Dataset_Reader:
    def __init__(self, data_name="gigaword", test_size=0.1, data_set_size=20000,mode="training"):
        """
        data set reader
        """
        if mode=="training":
            dataset=load_dataset(data_name,split = 'train')
            train_data, test_data= dataset.train_test_split(test_size=test_size).values()
            small_dataset= datasets.DatasetDict({'train':train_data,'test':test_data})
            self.input_sentences=small_dataset['train']['document'][:data_set_size]
            self.output_sentences=small_dataset['train']['summary'][:data_set_size]
            self.input_test=small_dataset['test']['document'][:data_set_size]
            self.output_test=small_dataset['test']['summary'][:data_set_size]

    def get_training(self):
        return self.input_sentences,self.output_sentences
    def get_testing(self):
        return self.input_test,self.output_test