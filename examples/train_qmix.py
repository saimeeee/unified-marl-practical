from unified_marl import train, evaluate


if __name__ == "__main__":
    result = train("pymarl/qmix", "toy", overrides={"batch_size": 2, "buffer_size": 8, "t_max": 32})
    print(result)
    print(evaluate(result.checkpoint, episodes=2))
