"""Run with an installed package; Windows multiprocessing requires this guard."""
from unified_marl import train,evaluate

def main():
    result=train("sadma/qmix","toy",execution="zmq",output_dir="./results/zmq-example",
        overrides={"train_batch_size":2,"train_buffer_size":8,"update_rate":1,"hidden_dim":16,
                   "t_max":40,"max_updates":2,"zmq.num_sampling_workers":2,"zmq.num_inference_workers":1,"zmq.mode":"async"})
    print(result)
    print(evaluate(result.checkpoint,episodes=2))

if __name__=="__main__": main()
