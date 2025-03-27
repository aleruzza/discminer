import tracemalloc

def measure_memory_usage(func):
    def wrapper(self, *args, **kwargs):
        tracemalloc.start()
        self.wrapper = "I'm a wrapper"
        # Call the original function
        result = func(self, *args, **kwargs)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Print the top memory-consuming lines
        print(f"Memory usage of {func.__name__}:")
        for stat in top_stats[:5]:
            print(stat)

        # Return the result
        return result

    return wrapper