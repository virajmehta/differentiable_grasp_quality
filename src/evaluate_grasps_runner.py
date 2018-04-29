from multiprocessing import Pool
import _init_paths
from lib.data_io import category_model_id_pair
from evaluate_grasps import evaluate_grasps

NUM_PROCESSES=6

def main():
    args = [(category, model_id) for category, model_id in
                    category_model_id_pair(dataset_portion=[0,1])]
    
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.starmap(evaluate_grasps, args)
        
main()        
