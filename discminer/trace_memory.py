import tracemalloc

def measure_memory_usage(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        # Call the original function
        result = func(*args, **kwargs)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Print the top memory-consuming lines
        print(f"Memory usage of {func.__name__}:")
        for stat in top_stats[:5]:
            print(stat)

        # Return the result
        return result

    return wrapper