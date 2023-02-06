import steps.data_understanding as du
import steps.data_preparation as dp
import steps.data_modeling as dm

du.data_understanding()
dataset = dp.data_preparation()
dm.data_modeling(dataset)
