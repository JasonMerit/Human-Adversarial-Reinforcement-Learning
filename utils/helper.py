class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def has_wrapper(env, wrapper_class):
    """Check if `env` or any of its wrappers is an instance of `wrapper_class`."""
    current = env
    while current:
        if isinstance(current, wrapper_class):
            return True
        # Stop if there is no further wrapped env
        if not hasattr(current, 'env'):
            break
        current = current.env
    return False
