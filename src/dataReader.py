import datasets


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
        if mode== "testing":
            train_ds, test_ds = datasets.load_dataset(data_name, split=['train', 'test'])
            ## todo:: filling this up
    def get_input(self):
        return self.input_sentences
    def get_output(self):
        return self.output_sentences